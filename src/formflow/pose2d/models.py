"""YOLOX person detector + RTMPose keypoint model (OpenMMLab ONNX exports, run with ONNX Runtime).

Pre/post-processing follows the ``pipeline.json`` shipped with each export:

YOLOX    keep-ratio resize to the input square, pad bottom/right with grey 114, BGR, no normalisation.
         The graph already contains NMS and outputs (x1, y1, x2, y2, score) + class labels.
RTMPose  top-down: the person box is padded 1.25x, fixed to the 192:256 aspect ratio and warped to
         192x256 RGB, ImageNet-normalised. SimCC heads give one 1-D distribution per axis per joint
         (2 bins per pixel); the keypoint is the argmax on each axis, its score the smaller peak.
"""

from __future__ import annotations

import cv2
import numpy as np

from .runtime import make_session, session_device


class YOLOX:
    def __init__(self, path: str, device: str = "auto", score_thr: float = 0.35) -> None:
        self.sess = make_session(path, device)
        self.input = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = (int(shape[2]), int(shape[3]))  # (h, w)
        self.score_thr = score_thr
        self.device = session_device(self.sess)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """Person boxes (N, 5): x1, y1, x2, y2, score - in image pixels, best first."""
        h, w = img_bgr.shape[:2]
        th, tw = self.size
        r = min(th / h, tw / w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((th, tw, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        dets, labels = self.sess.run(None, {self.input: x})
        dets, labels = dets[0], labels[0]
        keep = (labels == 0) & (dets[:, 4] >= self.score_thr)
        dets = dets[keep].copy()
        dets[:, :4] /= r
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, w)
        dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, h)
        return dets[np.argsort(-dets[:, 4])]


class RTMPose:
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, path: str, device: str = "auto", padding: float = 1.25) -> None:
        self.sess = make_session(path, device)
        self.input = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = (int(shape[3]), int(shape[2]))  # (w, h) = (192, 256)
        self.padding = padding
        self.device = session_device(self.sess)

    def _crop(self, img: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = box[:4]
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
        scale = np.array([x2 - x1, y2 - y1], np.float32) * self.padding
        aspect = self.size[0] / self.size[1]
        if scale[0] > scale[1] * aspect:
            scale[1] = scale[0] / aspect
        else:
            scale[0] = scale[1] * aspect
        s = self.size[0] / max(scale[0], 1e-3)
        M = np.array([[s, 0, self.size[0] / 2 - s * center[0]], [0, s, self.size[1] / 2 - s * center[1]]], np.float32)
        crop = cv2.warpAffine(img, M, self.size, flags=cv2.INTER_LINEAR)
        return crop, center, scale

    def __call__(self, img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Keypoints (N, 17, 3): x, y in image pixels and a confidence score, one row per box."""
        if len(boxes) == 0:
            return np.zeros((0, 17, 3), np.float32)
        crops, centers, scales = [], [], []
        for b in boxes:
            c, ce, sc = self._crop(img_bgr, b)
            crops.append(c)
            centers.append(ce)
            scales.append(sc)
        x = np.stack(crops)[..., ::-1].astype(np.float32)  # BGR -> RGB
        x = ((x - self.MEAN) / self.STD).transpose(0, 3, 1, 2)
        sx, sy = self.sess.run(None, {self.input: np.ascontiguousarray(x)})
        locs, vals = decode_simcc(sx, sy, split_ratio=2.0)
        centers, scales = np.stack(centers), np.stack(scales)
        size = np.array(self.size, np.float32)
        xy = locs / size * scales[:, None] + centers[:, None] - scales[:, None] / 2
        return np.concatenate([xy, vals[..., None]], axis=-1).astype(np.float32)


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """(N, K, Wx), (N, K, Wy) -> locations (N, K, 2) in model-input pixels, scores (N, K)."""
    xi = simcc_x.argmax(-1)
    yi = simcc_y.argmax(-1)
    locs = np.stack([xi, yi], -1).astype(np.float32) / split_ratio
    vals = np.minimum(simcc_x.max(-1), simcc_y.max(-1))
    return locs, vals
