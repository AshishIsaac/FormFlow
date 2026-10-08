"""Biomechanical measurements from a 3D H36M skeleton.

Coordinates are the lifter's camera frame (x right, y down, z away from the camera), root-relative.
"Up" is the direction against gravity; by default it is the camera's -y axis, refined from the
standing pose when the exercise is a standing one (see ``UpEstimator``), so a phone propped at an
angle still gives true trunk inclinations.

Every function accepts (17, 3) or (T, 17, 3) and returns a float or a (T,) array. Angles are degrees,
with 0 meaning "straight / neutral": knee flexion 0 = locked knee, trunk inclination 0 = upright.
"""

from __future__ import annotations

import numpy as np

from ..skeleton import J

CAMERA_UP = np.array([0.0, -1.0, 0.0])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    c = np.sum(_unit(u) * _unit(v), axis=-1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def joint_angle(p: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Interior angle at joint ``b`` between segments b->a and b->c (180 = straight line)."""
    return angle_between(p[..., a, :] - p[..., b, :], p[..., c, :] - p[..., b, :])


def flexion(p: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """180 - interior angle: 0 when the limb is straight, larger when it bends."""
    return 180.0 - joint_angle(p, a, b, c)


def mid(p: np.ndarray, a: int, b: int) -> np.ndarray:
    return 0.5 * (p[..., a, :] + p[..., b, :])


# --------------------------------------------------------------------------- body frame


def leg_length(p: np.ndarray) -> np.ndarray:
    seg = lambda a, b: np.linalg.norm(p[..., a, :] - p[..., b, :], axis=-1)
    return 0.5 * (seg(J["l_hip"], J["l_knee"]) + seg(J["l_knee"], J["l_ankle"])
                  + seg(J["r_hip"], J["r_knee"]) + seg(J["r_knee"], J["r_ankle"]))


def lateral_axis(p: np.ndarray) -> np.ndarray:
    """Unit vector from the right hip to the left hip (the body's left)."""
    return _unit(p[..., J["l_hip"], :] - p[..., J["r_hip"], :])


def forward_axis(p: np.ndarray, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Direction the pelvis faces, horizontal: left x up (facing the camera gives -z in the y-down frame)."""
    up = np.broadcast_to(up, p[..., 0, :].shape)
    left = lateral_axis(p)
    f = np.cross(left, up)
    return _unit(f - np.sum(f * up, axis=-1, keepdims=True) * up)


class UpEstimator:
    """Running estimate of gravity's direction for standing exercises.

    When the athlete stands tall (knees and hips near straight), the ankle-midpoint -> thorax line is
    close to vertical. Blending that observation into the camera's up axis corrects for a tilted camera.
    The correction is limited to ``max_tilt`` degrees so a lean or a stumble can't flip the world.
    """

    def __init__(self, max_tilt: float = 30.0, rate: float = 0.02) -> None:
        self.up = CAMERA_UP.copy()
        self.max_tilt = max_tilt
        self.rate = rate
        self.samples = 0

    def update(self, p: np.ndarray) -> np.ndarray:
        knee = max(float(knee_flexion(p, "l")), float(knee_flexion(p, "r")))
        hip = max(float(hip_flexion(p, "l")), float(hip_flexion(p, "r")))
        if knee < 15 and hip < 15:
            v = _unit(p[J["thorax"]] - mid(p, J["l_ankle"], J["r_ankle"]))
            if angle_between(v, CAMERA_UP) <= self.max_tilt:
                self.up = _unit((1 - self.rate) * self.up + self.rate * v)
                self.samples += 1
        return self.up


# --------------------------------------------------------------------------- measurements


def knee_flexion(p: np.ndarray, side: str) -> np.ndarray:
    s = side[0]
    return flexion(p, J[f"{s}_hip"], J[f"{s}_knee"], J[f"{s}_ankle"])


def hip_flexion(p: np.ndarray, side: str) -> np.ndarray:
    """Angle between the trunk (pelvis -> thorax) and the thigh (knee -> hip): 0 standing tall, ~90 seated.

    The trunk line runs through the pelvis centre rather than the side hip joint, so the hips' lateral
    offset does not add a constant ~15 degrees.
    """
    s = side[0]
    trunk = p[..., J["thorax"], :] - p[..., J["pelvis"], :]
    thigh_up = p[..., J[f"{s}_hip"], :] - p[..., J[f"{s}_knee"], :]
    return angle_between(trunk, thigh_up)


def elbow_flexion(p: np.ndarray, side: str) -> np.ndarray:
    s = side[0]
    return flexion(p, J[f"{s}_shoulder"], J[f"{s}_elbow"], J[f"{s}_wrist"])


def arm_elevation(p: np.ndarray, side: str) -> np.ndarray:
    """Angle of the upper arm away from the trunk's downward line: 0 arms by the side, 180 overhead."""
    s = side[0]
    down = p[..., J["pelvis"], :] - p[..., J["thorax"], :]
    arm = p[..., J[f"{s}_elbow"], :] - p[..., J[f"{s}_shoulder"], :]
    return angle_between(arm, down)


def trunk_inclination(p: np.ndarray, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Angle of the pelvis -> thorax line from vertical: 0 upright, 90 horizontal."""
    return angle_between(p[..., J["thorax"], :] - p[..., J["pelvis"], :], np.broadcast_to(up, p[..., 0, :].shape))


def spine_flexion(p: np.ndarray) -> np.ndarray:
    """Bend of the trunk at the mid-spine joint: angle between pelvis -> spine and spine -> thorax.

    A neutral spine reads 5-15 degrees; a rounded (flexed) back clearly more. The 2D input has no spine
    keypoint (it is the hip-shoulder midpoint), so this comes from the lifter's learned body prior; the
    neck/head joints are deliberately left out because head nods would read as a rounded back.
    """
    return angle_between(p[..., J["spine"], :] - p[..., J["pelvis"], :], p[..., J["thorax"], :] - p[..., J["spine"], :])


def knee_valgus(p: np.ndarray, side: str, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Frontal-plane projection angle: how far the knee falls inside the hip -> ankle line (degrees).

    Positive = valgus (knee caving in), negative = varus. The three joints are projected onto the
    body's frontal plane (spanned by the left axis and up) before measuring.
    """
    s = side[0]
    hip, knee, ankle = p[..., J[f"{s}_hip"], :], p[..., J[f"{s}_knee"], :], p[..., J[f"{s}_ankle"], :]
    fwd = forward_axis(p, up)

    def proj(v):
        return v - np.sum(v * fwd, axis=-1, keepdims=True) * fwd

    th, sh = proj(knee - hip), proj(ankle - knee)
    ang = angle_between(th, sh)
    # Sign: the knee sits medially if it is displaced toward the body's midline relative to hip->ankle.
    left = lateral_axis(p)
    line = proj(ankle - hip)
    t = np.sum(proj(knee - hip) * _unit(line), axis=-1, keepdims=True)
    off = proj(knee - hip) - t * _unit(line)
    medial = -np.sum(off * left, axis=-1) if s == "l" else np.sum(off * left, axis=-1)
    return np.where(medial >= 0, ang, -ang)


def knee_medial(p: np.ndarray, side: str, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Knee displacement toward the midline from the hip -> ankle line, in % of leg length (frontal plane).

    Better conditioned than the projection angle in deep flexion (the projected thigh gets short
    there, so a small drift reads as a huge angle). Positive = valgus.
    """
    s = side[0]
    hip, knee, ankle = p[..., J[f"{s}_hip"], :], p[..., J[f"{s}_knee"], :], p[..., J[f"{s}_ankle"], :]
    fwd = forward_axis(p, up)

    def proj(v):
        return v - np.sum(v * fwd, axis=-1, keepdims=True) * fwd

    line = _unit(proj(ankle - hip))
    rel = proj(knee - hip)
    off = rel - np.sum(rel * line, axis=-1, keepdims=True) * line
    left = lateral_axis(p)
    medial = -np.sum(off * left, axis=-1) if s == "l" else np.sum(off * left, axis=-1)
    return 100.0 * medial / np.maximum(leg_length(p), 1e-6)


def knee_ankle_ratio(p: np.ndarray) -> np.ndarray:
    """Knee separation / ankle separation (KASR). Below ~0.8 the knees are collapsing inward."""
    k = np.linalg.norm(p[..., J["l_knee"], :] - p[..., J["r_knee"], :], axis=-1)
    a = np.linalg.norm(p[..., J["l_ankle"], :] - p[..., J["r_ankle"], :], axis=-1)
    return k / np.maximum(a, 1e-6)


def hip_depth(p: np.ndarray, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Height of the hips above the knees along gravity, in thigh lengths. <= 0 means at or below parallel."""
    up = np.broadcast_to(up, p[..., 0, :].shape)
    hips = mid(p, J["l_hip"], J["r_hip"])
    knees = mid(p, J["l_knee"], J["r_knee"])
    thigh = 0.5 * (np.linalg.norm(p[..., J["l_hip"], :] - p[..., J["l_knee"], :], axis=-1)
                   + np.linalg.norm(p[..., J["r_hip"], :] - p[..., J["r_knee"], :], axis=-1))
    return np.sum((hips - knees) * up, axis=-1) / np.maximum(thigh, 1e-6)


def body_line(p: np.ndarray, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Plank line shoulder -> hip -> ankle, as signed deviation from straight (degrees).

    Positive = hips sagging (below the shoulder-ankle line), negative = hips piked up.
    """
    sh = p[..., J["thorax"], :]
    hip = p[..., J["pelvis"], :]
    an = mid(p, J["l_ankle"], J["r_ankle"])
    dev = 180.0 - angle_between(sh - hip, an - hip)
    line = an - sh
    t = np.sum((hip - sh) * _unit(line), axis=-1, keepdims=True)
    off = (hip - sh) - t * _unit(line)
    below = -np.sum(off * np.broadcast_to(up, off.shape), axis=-1)  # >0: hip lower than the line
    return np.where(below >= 0, dev, -dev)


def wrist_height(p: np.ndarray, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Mean wrist height above the head, in forearm+upper-arm lengths (pull-up: > 0 = hands above head)."""
    up = np.broadcast_to(up, p[..., 0, :].shape)
    w = mid(p, J["l_wrist"], J["r_wrist"])
    arm = np.linalg.norm(p[..., J["l_shoulder"], :] - p[..., J["l_elbow"], :], axis=-1) + np.linalg.norm(
        p[..., J["l_elbow"], :] - p[..., J["l_wrist"], :], axis=-1)
    return np.sum((w - p[..., J["head"], :]) * up, axis=-1) / np.maximum(arm, 1e-6)


def head_over_hands(p: np.ndarray, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Pull-up: height of the head above the wrists, in head-to-thorax lengths (> 0 = chin over the bar)."""
    up = np.broadcast_to(up, p[..., 0, :].shape)
    w = mid(p, J["l_wrist"], J["r_wrist"])
    unit = np.linalg.norm(p[..., J["head"], :] - p[..., J["thorax"], :], axis=-1)
    return np.sum((p[..., J["head"], :] - w) * up, axis=-1) / np.maximum(unit, 1e-6)


def stance_ratio(p: np.ndarray) -> np.ndarray:
    """Ankle separation / hip width (jumping jacks: feet apart vs together)."""
    a = np.linalg.norm(p[..., J["l_ankle"], :] - p[..., J["r_ankle"], :], axis=-1)
    h = np.linalg.norm(p[..., J["l_hip"], :] - p[..., J["r_hip"], :], axis=-1)
    return a / np.maximum(h, 1e-6)


def pelvis_height(p: np.ndarray, up: np.ndarray = CAMERA_UP) -> np.ndarray:
    """Pelvis height above the ankles along gravity, in leg lengths (1 standing, ~0.5 deep squat)."""
    up = np.broadcast_to(up, p[..., 0, :].shape)
    return np.sum((p[..., J["pelvis"], :] - mid(p, J["l_ankle"], J["r_ankle"])) * up, axis=-1) / np.maximum(
        leg_length(p), 1e-6)


# --------------------------------------------------------------------------- the metric table

_SIDED = {
    "knee_flexion": knee_flexion,
    "hip_flexion": hip_flexion,
    "elbow_flexion": elbow_flexion,
    "arm_elevation": arm_elevation,
}


def compute_metrics(p: np.ndarray, up: np.ndarray = CAMERA_UP) -> dict[str, np.ndarray]:
    """All named metrics for one frame (17, 3) or a sequence (T, 17, 3).

    Sided metrics come as ``<name>_l``, ``<name>_r``, their mean ``<name>``, the larger ``<name>_max``
    and the left/right difference ``<name>_asym`` (absolute, degrees).
    """
    m: dict[str, np.ndarray] = {}
    for name, fn in _SIDED.items():
        left, right = fn(p, "l"), fn(p, "r")
        m[f"{name}_l"], m[f"{name}_r"] = left, right
        m[name] = 0.5 * (left + right)
        m[f"{name}_max"] = np.maximum(left, right)
        m[f"{name}_asym"] = np.abs(left - right)
    vl, vr = knee_valgus(p, "l", up), knee_valgus(p, "r", up)
    m["knee_valgus_l"], m["knee_valgus_r"], m["knee_valgus_max"] = vl, vr, np.maximum(vl, vr)
    ml, mr = knee_medial(p, "l", up), knee_medial(p, "r", up)
    m["knee_medial_l"], m["knee_medial_r"], m["knee_medial_max"] = ml, mr, np.maximum(ml, mr)
    m["knee_ankle_ratio"] = knee_ankle_ratio(p)
    m["trunk_inclination"] = trunk_inclination(p, up)
    m["spine_flexion"] = spine_flexion(p)
    m["hip_depth"] = hip_depth(p, up)
    m["body_line"] = body_line(p, up)
    m["wrist_height"] = wrist_height(p, up)
    m["head_over_hands"] = head_over_hands(p, up)
    m["stance_ratio"] = stance_ratio(p)
    m["pelvis_height"] = pelvis_height(p, up)
    # How side-on the camera sees the athlete: |hip line . viewing direction| (0 = facing, 1 = side view).
    # Left/right comparisons and frontal-plane angles are unreliable when this is high (one side is hidden).
    m["side_view"] = np.abs(lateral_axis(p)[..., 2])
    return m


METRIC_UNITS = {
    "knee_medial_l": "%leg", "knee_medial_r": "%leg", "knee_medial_max": "%leg", "knee_ankle_ratio": "x", "hip_depth": "thigh", "wrist_height": "arm", "head_over_hands": "head",
    "stance_ratio": "x", "pelvis_height": "leg",
}


def unit_of(metric: str) -> str:
    return METRIC_UNITS.get(metric, "deg")
