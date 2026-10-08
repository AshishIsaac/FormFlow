"""Analyse a recorded video: reps, per-rep scores and faults, an annotated video and a report.

Offline analysis sees the whole clip, so the lifter gets past *and* future frames (243-frame windows,
50% overlap, flip augmentation) - more accurate than the causal live path. ``mode="live"`` replays the
video through the exact live pipeline instead (used to benchmark what the live coach would have said).

Outputs in ``<out_dir>``:
    annotated.mp4   the video with the 2D skeleton, 3D view, rep counter, cues and signal strip
    report.json     summary, every rep with its faults, and the per-frame rep signal
    report.html     the same as a self-contained page
    tracks.npz      2D keypoints, 3D poses and every joint metric per frame
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .coach.exercises import Exercise, get_exercise
from .lift.lifter import StreamingLifter
from .pipeline import Coach, CoachUpdate
from .render.overlay import Renderer, angle_keys_for
from .signal.one_euro import OneEuroFilter
from .skeleton import coco_to_h36m

Progress = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    pass


def read_video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps < 1 or fps > 240:
        fps = 30.0
    info = {"fps": float(fps), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def iter_frames(path: str | Path, max_side: int | None = None):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if max_side and max(img.shape[:2]) > max_side:
                s = max_side / max(img.shape[:2])
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            yield img
    finally:
        cap.release()


def track_2d(engine, path: str | Path, cfg: dict, progress: Progress = _noop, max_side: int | None = None,
             max_frames: int | None = None) -> tuple[np.ndarray, list, tuple[int, int], float]:
    info = read_video_info(path)
    tracker = engine.tracker(det_every=int(cfg.get("offline", {}).get("det_every", 5)))
    kps, boxes = [], []
    wh = None
    n = max(info["frames"], 1)
    for i, img in enumerate(iter_frames(path, max_side)):
        if max_frames and i >= max_frames:
            break
        wh = (img.shape[1], img.shape[0])
        p = tracker(img)
        kps.append(p.keypoints)
        boxes.append(None if p.box is None else p.box.tolist())
        if i % 15 == 0:
            progress("2D keypoints", min(i / n, 1.0))
    if wh is None:
        raise ValueError(f"no frames could be read from {path}")
    return np.stack(kps).astype(np.float32), boxes, wh, info["fps"]


def smooth_2d(kp: np.ndarray, fps: float, cfg: dict, mode: str = "offline") -> np.ndarray:
    """One Euro filter on the 2D keypoints - live mode only by default.

    Offline the lifter already sees future frames and its output is smooth; the causal filter would
    only add lag (H36M: 8.40 -> 8.56 deg of joint-angle error). ``offline.one_euro_2d: true`` turns it on.
    """
    if mode == "offline" and not cfg.get("offline", {}).get("one_euro_2d", False):
        return kp
    if mode == "live" and not cfg.get("live", {}).get("one_euro_2d_lift", False):
        return kp
    f = OneEuroFilter(**cfg.get("one_euro", {}))
    out = kp.copy()
    for i in range(len(kp)):
        if kp[i, :, 2].mean() > 0.2:
            out[i, :, :2] = f(kp[i, :, :2], i / fps, conf=kp[i, :, 2])
    return out


def lift(engine, kp_coco: np.ndarray, wh: tuple[int, int], cfg: dict, mode: str = "offline", fps: float = 30.0) -> np.ndarray:
    h36 = coco_to_h36m(kp_coco)
    if mode == "offline":
        o = cfg.get("offline", {})
        engine.lifter.flip = bool(o.get("flip", True))
        engine.lifter.half = bool(o.get("half", False)) and engine.lifter.device == "cuda"
        return engine.lifter.lift_sequence(h36, wh, window=int(o.get("window", 243)), overlap=float(o.get("overlap", 0.5)))
    from .config import live_settings

    lv = live_settings(cfg, engine.lifter.device)
    engine.lifter.flip = bool(lv.get("flip", False))
    engine.lifter.half = bool(lv.get("half", True)) and engine.lifter.device == "cuda"
    stream = StreamingLifter(engine.lifter, wh, window=int(lv.get("window", 81)), stride=int(lv.get("stride", 2)))
    out = np.zeros((len(h36), 17, 3), np.float32)
    for i, k in enumerate(h36):
        for f, pose in stream.push(i, k, force=(i == len(h36) - 1)):
            out[f] = pose
    if lv.get("one_euro_3d"):
        f3 = OneEuroFilter(**lv["one_euro_3d"])
        out = np.stack([f3(p, i / fps) for i, p in enumerate(out)]).astype(np.float32)
    return out


def coach_sequence(ex: Exercise, poses: np.ndarray, fps: float, visible: np.ndarray | None = None,
                   up_mode: str = "auto") -> tuple[Coach, list[CoachUpdate]]:
    coach = Coach(ex, fps=fps, up_mode=up_mode)
    ups = []
    for i, p in enumerate(poses):
        if visible is not None and not visible[i]:
            prev = ups[-1] if ups else CoachUpdate()
            ups.append(CoachUpdate(**{**prev.__dict__, "reps": [], "partials": [], "cues": [], "live": []}))
            continue
        ups.append(coach.ingest(i, i / fps, p))
    tail = coach.finish()
    if tail and ups:
        ups[-1].reps += tail
    return coach, ups


def analyze_video(path: str | Path, exercise: str | Exercise, cfg: dict, engine=None, out_dir: Path | None = None,
                  render: bool = True, mode: str = "offline", progress: Progress = _noop,
                  max_frames: int | None = None) -> dict:
    from .engine import build_engine

    t0 = time.time()
    ex = exercise if isinstance(exercise, Exercise) else get_exercise(exercise)
    engine = engine or build_engine(cfg, "offline" if mode == "offline" else "live")
    max_side = int(cfg.get("live", {}).get("max_side", 960))
    progress("2D keypoints", 0.0)
    kp, boxes, wh, fps = track_2d(engine, path, cfg, progress, max_side=max_side, max_frames=max_frames)
    t2d = time.time() - t0
    kp_s = smooth_2d(kp, fps, cfg, mode)
    visible = kp[:, :, 2].mean(1) > 0.2
    progress("3D lifting", 0.0)
    t1 = time.time()
    poses = lift(engine, kp_s, wh, cfg, mode, fps)
    t3d = time.time() - t1
    progress("coaching", 0.0)
    coach, ups = coach_sequence(ex, poses, fps, visible, cfg.get("up", "auto"))
    summary = coach.summary()
    n = len(kp)
    result = {
        "video": str(path), "exercise": ex.name, "display": ex.display, "mode": mode, "fps": fps, "frames": n,
        "size": list(wh), "visible_frac": round(float(visible.mean()), 3), "summary": summary,
        "counts_by_signal": coach.counts_by_signal(),
        "timing": {"pose2d_s": round(t2d, 2), "lift_s": round(t3d, 2), "pose2d_ms_per_frame": round(1000 * t2d / max(n, 1), 2),
                   "lift_ms_per_frame": round(1000 * t3d / max(n, 1), 2)},
        "models": engine.describe(),
        "signal": {"name": summary["signal"], "values": [round(float(u.signal), 2) if u.signal is not None else None for u in ups],
                   "phase": [u.phase for u in ups]},
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        keys = sorted(ups[-1].metrics) if ups and ups[-1].metrics else []
        metrics = np.array([[u.metrics.get(k, np.nan) for k in keys] for u in ups], np.float32) if keys else np.zeros((n, 0))
        np.savez_compressed(out_dir / "tracks.npz", kp2d=kp, kp2d_smooth=kp_s, pose3d=poses, metrics=metrics,
                            metric_names=np.array(keys), fps=fps)
        if render:
            progress("rendering", 0.0)
            write_annotated(path, out_dir / "annotated.mp4", ex, kp_s, ups, fps, max_side, progress)
            result["annotated"] = "annotated.mp4"
        (out_dir / "report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        from .render.report import write_html

        write_html(result, out_dir / "report.html")
    result["elapsed_s"] = round(time.time() - t0, 2)
    progress("done", 1.0)
    return result


class VideoWriter:
    """H.264 MP4 through imageio-ffmpeg (plays in every browser); falls back to OpenCV's mp4v."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.size = size
        self._gen = None
        self._cv = None
        try:
            import imageio_ffmpeg

            self._gen = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", pix_fmt_in="bgr24",
                                                    output_params=["-crf", "23", "-preset", "veryfast",
                                                                   "-movflags", "+faststart"],
                                                    macro_block_size=2, ffmpeg_log_level="error")
            self._gen.send(None)
        except Exception:
            self._gen = None
            self._cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, img: np.ndarray) -> None:
        if img.shape[1] != self.size[0] or img.shape[0] != self.size[1]:
            img = cv2.resize(img, self.size)
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(img))
        else:
            self._cv.write(img)

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
        if self._cv is not None:
            self._cv.release()


def write_annotated(path, out: Path, ex: Exercise, kp: np.ndarray, ups: list[CoachUpdate], fps: float,
                    max_side: int | None, progress: Progress = _noop, height: int = 720) -> None:
    r = Renderer(ex.display, angle_keys_for(ex), target_h=height)
    writer = None
    n = len(ups)
    for i, img in enumerate(iter_frames(path, max_side)):
        if i >= n:
            break
        t = i / fps
        u = ups[i]
        for rep in u.reps:
            r.note_rep(t, rep)
        for c in u.cues:
            r.note_cue(t, c)
        r.yaw = -60.0 + 25.0 * np.sin(t * 0.4)
        frame = r.render(img, kp[i] if kp[i, :, 2].mean() > 0.2 else None, u, t)
        if writer is None:
            h, w = frame.shape[:2]
            writer = VideoWriter(out, fps, (w - w % 2, h - h % 2))
        writer.write(frame)
        if i % 30 == 0:
            progress("rendering", i / max(n, 1))
    if writer is not None:
        writer.close()
