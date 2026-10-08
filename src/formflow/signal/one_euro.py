"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012), vectorised over joints and coordinates.

A low-pass filter whose cutoff rises with speed: when a joint is still, a low cutoff removes jitter;
when it moves fast, the cutoff opens up so the filtered joint does not lag behind. ``beta`` sets how
quickly the cutoff opens. Timestamps (not frame indices) drive it, so dropped webcam frames are fine.

Low-confidence keypoints are handled by shrinking alpha for them: a joint the 2D model is unsure of
mostly keeps its previous filtered position instead of jumping to the new (probably wrong) one.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0,
                 conf_floor: float = 0.3) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.conf_floor = conf_floor
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x: np.ndarray, t: float, conf: np.ndarray | None = None) -> np.ndarray:
        """Filter one sample. ``x`` is (..., D); ``conf`` (...,) in [0, 1] optionally down-weights updates."""
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or t <= self._t:
            if self._x is not None and self._t is not None and t <= self._t:
                t = self._t + 1e-3  # duplicate timestamp: treat as a very short step
            else:
                self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
                return x.copy()
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        if conf is not None:
            w = np.clip((np.asarray(conf) - self.conf_floor) / max(1 - self.conf_floor, 1e-6), 0.05, 1.0)
            a = a * w[..., None]
        x_hat = a * x + (1 - a) * self._x
        self._x, self._dx, self._t = x_hat, dx_hat, t
        return x_hat.copy()


def filter_sequence(seq: np.ndarray, fps: float, conf: np.ndarray | None = None, **kw) -> np.ndarray:
    """Offline convenience: run the filter over a (T, ..., D) sequence sampled at ``fps``."""
    f = OneEuroFilter(**kw)
    out = np.empty_like(seq, dtype=np.float64)
    for i in range(len(seq)):
        out[i] = f(seq[i], i / fps, None if conf is None else conf[i])
    return out
