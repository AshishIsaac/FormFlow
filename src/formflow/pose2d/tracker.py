"""Follow one athlete through a video with a detector that runs only when needed.

The detector is the expensive part of top-down pose estimation (YOLOX-m: ~60 ms on a laptop RTX 3050,
RTMPose: ~7 ms). After the first detection the athlete's box for the next frame comes from the current
keypoints (their bounding box, padded), so the detector only re-runs every ``det_every`` frames or when
the pose looks lost. Among several people the one kept is the one overlapping the previous box most
(at the start: the largest, most central person).

With ``async_detect`` (live mode) the scheduled re-detections run on a background thread and are
applied when they finish, so they never stall a frame; only start-up and a lost track wait for one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .models import YOLOX, RTMPose


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def box_from_keypoints(kp: np.ndarray, img_wh: tuple[int, int], thr: float = 0.3, pad: float = 0.12) -> np.ndarray | None:
    good = kp[:, 2] >= thr
    if good.sum() < 6:
        return None
    xy = kp[good, :2]
    x1, y1 = xy.min(0)
    x2, y2 = xy.max(0)
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - pad * w, x2 + pad * w
    y1, y2 = y1 - pad * h - 0.08 * h, y2 + pad * h  # extra room above the nose for the top of the head
    W, H = img_wh
    return np.array([max(x1, 0), max(y1, 0), min(x2, W), min(y2, H), 1.0], np.float32)


@dataclass
class Pose2D:
    keypoints: np.ndarray  # (17, 3) COCO x, y, score - zeros when nobody is visible
    box: np.ndarray | None
    detected: bool  # the detector ran on this frame

    @property
    def visible(self) -> bool:
        return self.box is not None and float(np.mean(self.keypoints[:, 2])) > 0.2


class PoseTracker:
    def __init__(self, detector: YOLOX, pose: RTMPose, det_every: int = 10, lost_thr: float = 0.35,
                 async_detect: bool = False) -> None:
        self.det = detector
        self.pose = pose
        self.det_every = det_every
        self.lost_thr = lost_thr
        self.async_detect = async_detect
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self._box: np.ndarray | None = None
        self._since_det = 0
        self._bg: threading.Thread | None = None
        self._bg_result: tuple[np.ndarray, tuple[int, int]] | None = None

    def _bg_detect(self, img: np.ndarray, wh: tuple[int, int]) -> None:
        boxes = self.det(img)
        with self._lock:
            self._bg_result = (boxes, wh)

    def _scheduled_detection(self, img: np.ndarray, wh: tuple[int, int]) -> bool:
        """Async mode: start a background detection / apply a finished one. Returns True if one was applied."""
        applied = False
        with self._lock:
            res, self._bg_result = self._bg_result, None
        if res is not None:
            picked = self._pick(res[0], res[1])
            if picked is not None and (self._box is None or _iou(picked, self._box) > 0.1):
                self._box = picked
            applied = True
        if self._since_det >= self.det_every and (self._bg is None or not self._bg.is_alive()):
            self._bg = threading.Thread(target=self._bg_detect, args=(img.copy(), wh), daemon=True)
            self._bg.start()
            self._since_det = 0
        return applied

    def _pick(self, boxes: np.ndarray, img_wh: tuple[int, int]) -> np.ndarray | None:
        if len(boxes) == 0:
            return None
        if self._box is not None:
            ious = [_iou(b, self._box) for b in boxes]
            if max(ious) > 0.2:
                return boxes[int(np.argmax(ious))]
        W, H = img_wh
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (W * H)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2 / W
        central = 1 - np.abs(cx - 0.5)
        return boxes[int(np.argmax(area * (0.5 + central) * boxes[:, 4]))]

    def __call__(self, img_bgr: np.ndarray) -> Pose2D:
        H, W = img_bgr.shape[:2]
        detected = False
        if self.async_detect and self._box is not None:
            detected = self._scheduled_detection(img_bgr, (W, H))
        elif self._box is None or self._since_det >= self.det_every:
            picked = self._pick(self.det(img_bgr), (W, H))
            detected = True
            self._since_det = 0
            if picked is not None:
                self._box = picked
            elif self._box is None:
                return Pose2D(np.zeros((17, 3), np.float32), None, True)
        self._since_det += 1
        kp = self.pose(img_bgr, self._box[None])[0]
        if float(np.mean(kp[:, 2])) < self.lost_thr and not detected:
            # Lost track: detect right away instead of waiting for the next scheduled detection.
            picked = self._pick(self.det(img_bgr), (W, H))
            detected = True
            self._since_det = 1
            if picked is None:
                self._box = None
                return Pose2D(np.zeros((17, 3), np.float32), None, True)
            self._box = picked
            kp = self.pose(img_bgr, self._box[None])[0]
        used = self._box
        nxt = box_from_keypoints(kp, (W, H))
        if nxt is not None:
            self._box = nxt
        return Pose2D(kp, used, detected)
