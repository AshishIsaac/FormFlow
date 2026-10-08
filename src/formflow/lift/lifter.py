"""2D keypoint sequences -> 3D skeletons with MotionBERT.

Two ways to normalise the 2D input before it reaches the network:

``image``  Human3.6M's own convention: x / W * 2 - 1, y / W * 2 - H / W. Exactly what the model was
           trained on, but the input scale then depends on how big the athlete is in the frame.
``crop``   Centre on the athlete's bounding box over the window and scale it to ``target`` units
           (matched to the H36M training framing), so a person far from a phone and one close to a
           webcam look alike to the network. Used by the app. On H36M it costs ~0.3 degrees of
           angle error at H36M's own framing, but is unaffected by zoom, where ``image`` degrades
           from 6.6 to 11.8 degrees (benchmarks/results/BENCHMARK.md).

The 3D output is in the same units as the normalised input, root-relative (pelvis at the origin).
Joint angles do not depend on scale, so the app never needs metric units; the benchmark converts to
millimetres with H36M's per-frame 2.5D factor.

Offline (``lift_sequence``) clips overlap and are blended with a Hann window. Live
(``StreamingLifter``) runs the network on the last ``window`` frames every ``stride`` frames and
finalises the newest ``stride`` poses: causal, no future frames needed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..skeleton import H36M_FLIP

# "crop" mode: the athlete's extent over the window maps to CROP_TARGET units, its centre to CROP_Y below
# the origin - both tuned on H36M (per-frame extent median 0.66, centre ~0.15 above the image centre).
CROP_TARGET = 0.85
CROP_Y = -0.15


@dataclass
class Norm:
    center: np.ndarray  # (2,) pixels
    scale: float  # pixels per normalised unit

    def denorm(self, pose: np.ndarray) -> np.ndarray:
        """Normalised 3D -> pixels (x, y in image coordinates, z in the same pixel scale)."""
        out = pose * self.scale
        out[..., :2] += self.center
        return out


def normalise(kp: np.ndarray, img_wh: tuple[int, int], mode: str = "crop", target: float = CROP_TARGET,
              y_offset: float = CROP_Y, conf_thr: float = 0.3) -> tuple[np.ndarray, Norm]:
    """kp (T, 17, 3) pixels + confidence -> network input (T, 17, 3) and the transform to undo it."""
    W, H = img_wh
    if mode == "image":
        norm = Norm(np.array([W / 2, H / 2], np.float32), W / 2)
    else:
        good = kp[..., 2] >= conf_thr
        xy = kp[..., :2][good] if good.sum() >= 4 else kp[..., :2].reshape(-1, 2)
        lo, hi = xy.min(0), xy.max(0)
        extent = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
        scale = extent / target
        center = (lo + hi) / 2 - np.array([0.0, y_offset * scale])
        norm = Norm(center.astype(np.float32), scale)
    x = kp.astype(np.float32).copy()
    x[..., :2] = (x[..., :2] - norm.center) / norm.scale
    return x, norm


def fill_missing(kp: np.ndarray, conf_thr: float = 0.05) -> np.ndarray:
    """Frames with no detection (all-zero confidence) copy the nearest detected frame, confidence kept at 0."""
    kp = kp.copy()
    seen = kp[..., 2].max(-1) > conf_thr
    if not seen.any():
        return kp
    idx = np.where(seen)[0]
    nearest = idx[np.clip(np.searchsorted(idx, np.arange(len(kp))), 0, len(idx) - 1)]
    prev = idx[np.clip(np.searchsorted(idx, np.arange(len(kp))) - 1, 0, len(idx) - 1)]
    use = np.where(np.abs(prev - np.arange(len(kp))) < np.abs(nearest - np.arange(len(kp))), prev, nearest)
    miss = ~seen
    kp[miss, :, :2] = kp[use[miss], :, :2]
    kp[miss, :, 2] = 0.0
    return kp


class Lifter:
    """Owns the network. ``device`` auto|cuda|cpu; ``half`` runs it in float16 on the GPU."""

    def __init__(self, checkpoint: str | None, variant: str = "full", device: str = "auto", flip: bool = True,
                 half: bool = False, norm_mode: str = "crop", model=None) -> None:
        """``model`` passes an already built DSTformer instead of a checkpoint (tests, experiments)."""
        import torch

        from .dstformer import build, load_checkpoint

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.variant = variant
        if model is None:
            model = load_checkpoint(build(variant), checkpoint)
        self.model = model.eval().to(device)
        self.flip = flip
        self.half = half and device == "cuda"
        self.norm_mode = norm_mode
        self.maxlen = self.model.maxlen

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Batch of normalised clips (B, T, 17, 3) -> (B, T, 17, 3), root-relative."""
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        if self.flip:
            f = t.clone()
            f[..., 0] *= -1
            f = f[:, :, H36M_FLIP]
            t = torch.cat([t, f], 0)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            y = self.model(t).float()
        if self.flip:
            B = len(x)
            yf = y[B:].clone()
            yf[..., 0] *= -1
            y = 0.5 * (y[:B] + yf[:, :, H36M_FLIP])
        y = y - y[:, :, :1]
        return y.cpu().numpy()

    # ------------------------------------------------------------------ offline
    def lift_sequence(self, kp: np.ndarray, img_wh: tuple[int, int], window: int = 243, overlap: float = 0.5,
                      batch: int = 8) -> np.ndarray:
        """Whole video (N, 17, 3) H36M-ordered pixels -> (N, 17, 3) 3D in pixel scale, root-relative."""
        n = len(kp)
        if n == 0:
            return np.zeros((0, 17, 3), np.float32)
        kp = fill_missing(kp)
        T = min(window, self.maxlen, n)
        step = max(1, int(T * (1 - overlap)))
        starts = list(range(0, max(n - T, 0) + 1, step))
        if starts[-1] + T < n:
            starts.append(n - T)
        out = np.zeros((n, 17, 3), np.float64)
        wsum = np.zeros(n, np.float64)
        w = np.hanning(T + 2)[1:-1] if T > 2 else np.ones(T)
        for i in range(0, len(starts), batch):
            chunk = starts[i:i + batch]
            xs, norms = [], []
            for s in chunk:
                x, nm = normalise(kp[s:s + T], img_wh, self.norm_mode)
                xs.append(x)
                norms.append(nm)
            ys = self.infer(np.stack(xs))
            for s, y, nm in zip(chunk, ys, norms):
                out[s:s + T] += w[:, None, None] * (y * nm.scale)
                wsum[s:s + T] += w
        return (out / np.maximum(wsum, 1e-9)[:, None, None]).astype(np.float32)


class StreamingLifter:
    """Causal live lifting: feed one 2D pose per frame, get finalised 3D poses back.

    Every ``stride`` frames the network sees the last ``window`` frames (the first frame repeated until
    the buffer fills) and the newest ``stride`` outputs are finalised. ``stride`` trades GPU time for
    up to ``stride - 1`` frames of extra latency on the 3D view; joint angles and reps use every frame.
    """

    def __init__(self, lifter: Lifter, img_wh: tuple[int, int], window: int = 81, stride: int = 2) -> None:
        self.lifter = lifter
        self.img_wh = img_wh
        self.window = min(window, lifter.maxlen)
        self.stride = max(1, min(stride, self.window))
        self.reset()

    def reset(self) -> None:
        self.buf: deque[np.ndarray] = deque(maxlen=self.window)
        self.frames: deque[int] = deque(maxlen=self.window)
        self.pending = 0

    def push(self, frame: int, kp_h36m: np.ndarray, force: bool = False) -> list[tuple[int, np.ndarray]]:
        """Add a frame's (17, 3) H36M 2D pose. Returns [(frame, (17, 3) pose in pixel scale)] now final."""
        kp = kp_h36m.astype(np.float32)
        if self.buf and kp[:, 2].max() <= 0.05:  # nobody detected: hold the last position at zero confidence
            kp = self.buf[-1].copy()
            kp[:, 2] = 0
        self.buf.append(kp)
        self.frames.append(frame)
        self.pending += 1
        if self.pending < self.stride and not force:
            return []
        clip = np.stack(self.buf)
        if len(clip) < self.window:
            clip = np.concatenate([np.repeat(clip[:1], self.window - len(clip), 0), clip])
        x, nm = normalise(clip, self.img_wh, self.lifter.norm_mode)
        y = self.lifter.infer(x[None])[0] * nm.scale
        k = min(self.pending, len(self.frames))
        self.pending = 0
        fr = list(self.frames)[-k:]
        return [(f, y[len(y) - k + i]) for i, f in enumerate(fr)]
