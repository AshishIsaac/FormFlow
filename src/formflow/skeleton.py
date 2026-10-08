"""Joint layouts: COCO-17 (what the 2D model predicts) and Human3.6M-17 (what the 3D lifter speaks).

The 2D -> H36M conversion follows MotionBERT's ``coco2h36m``: the pelvis, thorax and spine are
synthesised from hip / shoulder midpoints, and the lifter then places them in 3D from what it
learned about real bodies (so the 3D spine can bend even though its 2D input is a straight midpoint).
"""

from __future__ import annotations

import numpy as np

COCO_NAMES = (
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)
H36M_NAMES = (
    "pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "spine", "thorax",
    "neck", "head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
)
J = {n: i for i, n in enumerate(H36M_NAMES)}  # H36M name -> index
C = {n: i for i, n in enumerate(COCO_NAMES)}  # COCO name -> index

H36M_BONES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
)
COCO_BONES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
)
# Side of each H36M joint for colouring: -1 right, +1 left, 0 centre.
H36M_SIDE = np.array([0, -1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1])
COCO_SIDE = np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
# Left/right swap permutation (for horizontal-flip test-time augmentation).
H36M_FLIP = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])


def coco_to_h36m(kp: np.ndarray) -> np.ndarray:
    """(..., 17, C) COCO keypoints (x, y[, conf]) -> (..., 17, C) H36M order, midpoints synthesised."""
    y = np.zeros_like(kp)
    y[..., 0, :] = (kp[..., 11, :] + kp[..., 12, :]) * 0.5
    y[..., 1, :] = kp[..., 12, :]
    y[..., 2, :] = kp[..., 14, :]
    y[..., 3, :] = kp[..., 16, :]
    y[..., 4, :] = kp[..., 11, :]
    y[..., 5, :] = kp[..., 13, :]
    y[..., 6, :] = kp[..., 15, :]
    y[..., 8, :] = (kp[..., 5, :] + kp[..., 6, :]) * 0.5
    y[..., 7, :] = (y[..., 0, :] + y[..., 8, :]) * 0.5
    y[..., 9, :] = kp[..., 0, :]
    y[..., 10, :] = (kp[..., 1, :] + kp[..., 2, :]) * 0.5
    y[..., 11, :] = kp[..., 5, :]
    y[..., 12, :] = kp[..., 7, :]
    y[..., 13, :] = kp[..., 9, :]
    y[..., 14, :] = kp[..., 6, :]
    y[..., 15, :] = kp[..., 8, :]
    y[..., 16, :] = kp[..., 10, :]
    return y


def h36m_to_coco(p: np.ndarray) -> np.ndarray:
    """Approximate inverse for drawing: (..., 17, C) H36M -> COCO (face points placed around the head)."""
    y = np.zeros_like(p)
    pairs = {5: 11, 6: 14, 7: 12, 8: 15, 9: 13, 10: 16, 11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3, 0: 9}
    for c, h in pairs.items():
        y[..., c, :] = p[..., h, :]
    head, nose = p[..., 10, :], p[..., 9, :]
    side = p[..., 11, :] - p[..., 14, :]  # left shoulder - right shoulder
    side = side * 0.12
    y[..., 1, :] = 0.5 * (head + nose) + side * 0.5
    y[..., 2, :] = 0.5 * (head + nose) - side * 0.5
    y[..., 3, :] = 0.5 * (head + nose) + side * 1.2
    y[..., 4, :] = 0.5 * (head + nose) - side * 1.2
    return y
