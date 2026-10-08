"""Joint-angle accuracy of the 2D -> 3D stage against Human3.6M motion capture (S9 + S11 test set).

Input is the standard H36M benchmark input: 2D keypoints from a stacked-hourglass detector with
confidences (MotionBERT's ``h36m_sh_conf_cam_source_final.pkl``), so the numbers include realistic
2D detection noise. Ground truth is the Vicon marker-based 3D skeleton.

Reported per configuration:
  MPJPE / PA-MPJPE (mm)    the usual 3D pose errors (protocol 1 / 2), averaged per action then overall
  angle MAE (degrees)      for every joint angle the coach uses, predicted vs mocap
  accel (mm/frame^2)       acceleration error, a jitter measure - what the One Euro filter targets
  ms / frame               lifter time per output frame on this machine

Offline configurations see the whole clip (past and future); ``causal`` ones only see the past
``window`` frames, exactly like the live app (``StreamingLifter``).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..lift.lifter import Lifter, normalise
from ..signal.angles import compute_metrics
from ..signal.one_euro import filter_sequence

CAMERA_RES = {"54138969": (1000, 1002), "55011271": (1000, 1000), "58860488": (1000, 1000), "60457274": (1000, 1002)}
ANGLES = ("knee_flexion_l", "knee_flexion_r", "hip_flexion_l", "hip_flexion_r", "elbow_flexion_l",
          "elbow_flexion_r", "arm_elevation_l", "arm_elevation_r", "trunk_inclination", "spine_flexion",
          "knee_valgus_l", "knee_valgus_r")
FPS = 50.0


@dataclass
class LiftConfig:
    name: str
    variant: str = "full"
    window: int = 243
    causal: bool = False
    stride: int = 1  # causal: frames finalised per network run
    overlap: float = 0.0  # offline: clip overlap (0 = the MotionBERT protocol's non-overlapping clips)
    flip: bool = True
    half: bool = False
    norm: str = "image"
    one_euro: dict | None = None  # filter the 2D input before lifting
    one_euro_3d: dict | None = None  # filter the 3D output (causal jitter)
    zoom: float = 1.0  # scale the 2D input about the image centre (framing robustness; MPJPE then meaningless)
    cameras: list[str] = field(default_factory=lambda: list(CAMERA_RES))
    max_videos: int | None = None


def load_h36m_test(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def videos(data: dict, cameras: list[str], max_videos: int | None = None) -> list[tuple[str, np.ndarray]]:
    src = data["source"]
    change = np.r_[0, np.flatnonzero(src[1:] != src[:-1]) + 1, len(src)]
    out = []
    for a, b in zip(change[:-1], change[1:]):
        if str(data["camera"][a]) in cameras:
            out.append((str(src[a]), np.arange(a, b)))
    return out[:max_videos] if max_videos else out


def mpjpe(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p - g, axis=-1).mean(-1)


def pa_mpjpe(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Error after the best similarity transform (rotation, scale, translation) per frame."""
    mp, mg = p.mean(1, keepdims=True), g.mean(1, keepdims=True)
    p0, g0 = p - mp, g - mg
    np_, ng = np.linalg.norm(p0, axis=(1, 2), keepdims=True), np.linalg.norm(g0, axis=(1, 2), keepdims=True)
    p0, g0 = p0 / np_, g0 / ng
    H = np.einsum("nji,njk->nik", p0, g0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = V @ U.transpose(0, 2, 1)
    neg = np.linalg.det(R) < 0
    V[neg, :, -1] *= -1
    s[neg, -1] *= -1
    R = V @ U.transpose(0, 2, 1)
    tr = s.sum(1)[:, None, None]
    aligned = tr * ng * np.einsum("nij,nkj->nki", R, p0) + mg
    return np.linalg.norm(aligned - g, axis=-1).mean(-1)


def accel_error(p: np.ndarray, g: np.ndarray) -> float:
    if len(p) < 3:
        return float("nan")
    ap = p[2:] - 2 * p[1:-1] + p[:-2]
    ag = g[2:] - 2 * g[1:-1] + g[:-2]
    return float(np.linalg.norm(ap - ag, axis=-1).mean())


def causal_lift(lifter: Lifter, kp: np.ndarray, img_wh: tuple[int, int], window: int, stride: int,
                batch: int | None = None) -> np.ndarray:
    """Batched equivalent of feeding ``StreamingLifter`` one frame at a time."""
    n = len(kp)
    T = min(window, lifter.maxlen)
    batch = batch or max(4, int(48 * 81 / T))  # ~constant GPU memory across window lengths
    ends = list(range(stride - 1, n, stride))
    if not ends or ends[-1] != n - 1:
        ends.append(n - 1)
    padded = np.concatenate([np.repeat(kp[:1], T - 1, 0), kp])  # frame i is padded[i + T - 1]
    out = np.zeros((n, 17, 3), np.float32)
    prev = -1
    jobs = []
    for e in ends:
        jobs.append((e, prev))
        prev = e
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        xs, norms = [], []
        for e, _ in chunk:
            x, nm = normalise(padded[e:e + T], img_wh, lifter.norm_mode)
            xs.append(x)
            norms.append(nm)
        ys = lifter.infer(np.stack(xs))
        for (e, p), y, nm in zip(chunk, ys, norms):
            k = e - p
            out[p + 1:e + 1] = y[T - k:] * nm.scale
    return out


def evaluate(cfg: LiftConfig, data: dict, checkpoint: str, device: str = "auto", log=print) -> dict:
    lifter = Lifter(checkpoint, cfg.variant, device=device, flip=cfg.flip, half=cfg.half, norm_mode=cfg.norm)
    per_action: dict[str, list[np.ndarray]] = defaultdict(list)
    per_action_pa: dict[str, list[np.ndarray]] = defaultdict(list)
    ang_err: dict[str, list[np.ndarray]] = defaultdict(list)
    accel = []
    frames = 0
    t_net = 0.0
    vids = videos(data, cfg.cameras, cfg.max_videos)
    for vi, (_src, idx) in enumerate(vids):
        cam = str(data["camera"][idx[0]])
        wh = CAMERA_RES[cam]
        kp = np.concatenate([data["joint_2d"][idx], data["confidence"][idx][..., None]], -1).astype(np.float32)
        if cfg.zoom != 1.0:
            c = np.array(wh, np.float32) / 2
            kp[..., :2] = (kp[..., :2] - c) * cfg.zoom + c
        if cfg.one_euro:
            kp[..., :2] = filter_sequence(kp[..., :2], FPS, conf=kp[..., 2], **cfg.one_euro).astype(np.float32)
        if lifter.device == "cuda":
            lifter.torch.cuda.synchronize()
        t0 = time.perf_counter()
        if cfg.causal:
            pred = causal_lift(lifter, kp, wh, cfg.window, cfg.stride)
        else:
            pred = lifter.lift_sequence(kp, wh, window=cfg.window, overlap=cfg.overlap)
        if lifter.device == "cuda":
            lifter.torch.cuda.synchronize()
        t_net += time.perf_counter() - t0
        if cfg.one_euro_3d:
            pred = filter_sequence(pred, FPS, **cfg.one_euro_3d).astype(np.float32)
        fac = data["factor"][idx][:, None, None]
        pred_mm = pred * fac
        gt = data["joints_25d"][idx].astype(np.float64)
        gt = gt - gt[:, :1]
        pred_mm = pred_mm - pred_mm[:, :1]
        act = str(data["action"][idx[0]])
        per_action[act].append(mpjpe(pred_mm, gt))
        per_action_pa[act].append(pa_mpjpe(pred_mm, gt))
        accel.append((accel_error(pred_mm, gt), len(idx)))
        mp, mg = compute_metrics(pred_mm), compute_metrics(gt)
        for a in ANGLES:
            ang_err[a].append(np.abs(mp[a] - mg[a]))
        frames += len(idx)
        if log and (vi + 1) % 40 == 0:
            log(f"  [{cfg.name}] {vi + 1}/{len(vids)} videos")
    act_m = {a: float(np.concatenate(v).mean()) for a, v in sorted(per_action.items())}
    act_pa = {a: float(np.concatenate(v).mean()) for a, v in sorted(per_action_pa.items())}
    angles = {}
    for a in ANGLES:
        e = np.concatenate(ang_err[a])
        angles[a] = {"mae": round(float(e.mean()), 2), "median": round(float(np.median(e)), 2),
                     "p90": round(float(np.percentile(e, 90)), 2)}
    acc = np.array([a for a, _ in accel])
    wts = np.array([n for _, n in accel], float)
    good = np.isfinite(acc)
    return {
        "config": cfg.__dict__,
        "videos": len(vids),
        "frames": frames,
        "mpjpe_mm": round(float(np.mean(list(act_m.values()))), 2),
        "pa_mpjpe_mm": round(float(np.mean(list(act_pa.values()))), 2),
        "accel_mm": round(float(np.average(acc[good], weights=wts[good])), 2),
        "angle_mae_deg": round(float(np.mean([v["mae"] for v in angles.values()])), 2),
        "angles": angles,
        "per_action_mpjpe": {k: round(v, 1) for k, v in act_m.items()},
        "ms_per_frame": round(1000 * t_net / max(frames, 1), 3),
        "device": lifter.device,
    }


def save(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lift_h36m.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
