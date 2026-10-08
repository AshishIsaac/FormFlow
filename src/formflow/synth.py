"""A synthetic athlete: kinematically built H36M skeletons doing squats with chosen faults.

Used by the tests (exact ground truth for angles, rep boundaries and faults), by ``formflow demo``
and by the web app's "demo athlete" (no camera needed). Lengths are millimetres; the output is in the
lifter's camera frame (x right, y down, z away from the camera), pelvis at the origin.

The squat: shins tilt forward and thighs back so the knee flexes by ``depth`` degrees at the bottom,
the trunk leans forward to keep balance (plus ``chest_drop`` extra), the lower back rounds by
``back_round`` degrees, and each knee drifts toward the midline by ``valgus`` degrees of
frontal-plane angle (approximately) - all scaled by the smooth 0..1 excursion of each rep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .skeleton import J

L = {"hip_w": 125.0, "thigh": 440.0, "shank": 420.0, "low_spine": 230.0, "up_spine": 250.0, "neck": 180.0,
     "head": 110.0, "sh_w": 170.0, "uarm": 280.0, "farm": 250.0}


@dataclass
class SquatSpec:
    reps: int = 5
    depth: float = 125.0  # knee flexion at the bottom (degrees); ~125 puts the thighs at parallel
    down_s: float = 1.2
    up_s: float = 1.0
    pause_s: float = 0.6  # standing between reps
    lead_s: float = 1.0  # standing before the first rep
    valgus: float = 0.0  # frontal-plane knee collapse at the bottom (degrees)
    chest_drop: float = 0.0  # extra forward trunk lean at the bottom (degrees)
    back_round: float = 0.0  # lumbar flexion added at the bottom (degrees)
    asym: float = 0.0  # extra knee flexion on the left side (degrees)
    yaw: float = 0.0  # camera viewpoint around the vertical axis (degrees, 0 = facing the camera)
    noise_mm: float = 0.0
    fps: float = 30.0
    seed: int = 0


def excursion_curve(spec: SquatSpec) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """0..1 depth per frame and the ground-truth (start, bottom, end) frame of every rep."""
    fps = spec.fps
    e = [np.zeros(int(spec.lead_s * fps))]
    bounds = []
    n0 = len(e[0])
    for _ in range(spec.reps):
        nd, nu, npz = int(spec.down_s * fps), int(spec.up_s * fps), int(spec.pause_s * fps)
        down = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nd, endpoint=False))
        up = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nu, endpoint=False))
        e += [down, up, np.zeros(npz)]
        bounds.append((n0, n0 + nd, n0 + nd + nu))
        n0 += nd + nu + npz
    return np.concatenate(e), bounds


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def squat_pose(e: float, spec: SquatSpec) -> np.ndarray:
    """One skeleton at excursion ``e`` in a body frame: x = athlete's left, y = up, z = forward."""
    p = np.zeros((17, 3))
    up, fwd, left = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    for side, sgn, extra in (("l", 1.0, spec.asym), ("r", -1.0, 0.0)):
        k = np.radians(5 + e * (spec.depth - 5 + extra))  # knee flexion
        a = 0.30 * k  # shin forward tilt (~37 degrees at a parallel squat)
        b = k - a  # thigh backward tilt
        v = np.radians(spec.valgus * e)
        ankle = sgn * (L["hip_w"] + 30) * left
        knee = ankle + L["shank"] * (np.cos(a) * up + np.sin(a) * fwd)
        hip_rel = L["thigh"] * (np.cos(b) * up - np.sin(b) * fwd)
        # Valgus: the knee drifts toward the midline relative to the hip-ankle line.
        knee = knee - sgn * left * (np.sin(v) * L["shank"])
        hip = knee + hip_rel + sgn * left * (np.sin(v) * L["shank"])
        p[J[f"{side}_ankle"]], p[J[f"{side}_knee"]], p[J[f"{side}_hip"]] = ankle, knee, hip
    # With asymmetry the two legs end at different hip heights: a tilted pelvis, as in a real shift.
    mid = 0.5 * (p[J["l_hip"]] + p[J["r_hip"]])
    p[J["pelvis"]] = mid
    thigh_back = np.radians(5 + e * (spec.depth - 5)) * 0.70
    lean = 0.55 * thigh_back + np.radians(spec.chest_drop * e)
    rnd = np.radians(4 + spec.back_round * e)
    d1 = np.cos(lean) * up + np.sin(lean) * fwd
    d2 = np.cos(lean + rnd) * up + np.sin(lean + rnd) * fwd
    p[J["spine"]] = mid + L["low_spine"] * d1
    p[J["thorax"]] = p[J["spine"]] + L["up_spine"] * d2
    d3 = np.cos(lean + rnd - np.radians(10)) * up + np.sin(lean + rnd - np.radians(10)) * fwd
    p[J["neck"]] = p[J["thorax"]] + L["neck"] * d3 + 40 * fwd
    p[J["head"]] = p[J["neck"]] + L["head"] * up
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        sh = p[J["thorax"]] + sgn * L["sh_w"] * left
        el = sh + L["uarm"] * (np.cos(np.radians(70)) * -up + np.sin(np.radians(70)) * fwd)
        wr = el + L["farm"] * (np.cos(np.radians(20)) * up + np.sin(np.radians(20)) * fwd) - sgn * 60 * left
        p[J[f"{side}_shoulder"]], p[J[f"{side}_elbow"]], p[J[f"{side}_wrist"]] = sh, el, wr
    return p


def squat_sequence(spec: SquatSpec) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """(T, 17, 3) skeletons in the camera frame and the ground-truth rep boundaries."""
    e, bounds = excursion_curve(spec)
    rng = np.random.default_rng(spec.seed)
    # body frame (x left, y up, z forward) -> camera (x right in image, y down, z away). Facing the camera
    # the athlete's left is image-right and their forward is toward the camera (-z).
    to_cam = np.diag([1.0, -1.0, -1.0]) @ _rot_y(spec.yaw)
    seq = np.stack([squat_pose(x, spec) @ to_cam.T for x in e])
    seq = seq - seq[:, :1]
    if spec.noise_mm:
        seq = seq + rng.normal(0, spec.noise_mm, seq.shape)
        seq = seq - seq[:, :1]
    return seq.astype(np.float32), bounds


def project(seq: np.ndarray, img_wh: tuple[int, int] = (1280, 720), dist_mm: float = 3500.0,
            focal: float = 900.0) -> np.ndarray:
    """Pinhole projection of camera-frame skeletons to COCO-like pixel keypoints in H36M order (T, 17, 3)."""
    W, H = img_wh
    p = seq.astype(np.float64).copy()
    p[..., 2] += dist_mm
    p[..., 1] += 300.0  # pelvis a little below the optical axis
    x = focal * p[..., 0] / p[..., 2] + W / 2
    y = focal * p[..., 1] / p[..., 2] + H / 2
    return np.stack([x, y, np.ones_like(x)], -1).astype(np.float32)


def squat_set(per_rep: list[dict], base: SquatSpec | None = None) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """A set where every rep has its own settings, e.g. [{}, {"valgus": 8}, {"depth": 80}]."""
    base = base or SquatSpec()
    seqs, bounds, n = [], [], 0
    for i, kw in enumerate(per_rep):
        spec = SquatSpec(**{**base.__dict__, "reps": 1, "lead_s": base.lead_s if i == 0 else 0.0,
                            "seed": base.seed + i, **kw})
        seq, b = squat_sequence(spec)
        seqs.append(seq)
        bounds += [(s + n, m + n, e + n) for s, m, e in b]
        n += len(seq)
    return np.concatenate(seqs), bounds
