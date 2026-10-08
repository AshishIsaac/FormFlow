"""``formflow demo``: a synthetic athlete squatting with scripted faults, coached and rendered.

Needs no camera, no models and no GPU - the 3D skeleton comes from ``synth`` - so it is the quickest
way to see the coach's rep counting, scoring, cues and overlays, and it doubles as a smoke test.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .analyze import VideoWriter
from .coach.exercises import get_exercise
from .paths import sessions_dir
from .pipeline import Coach
from .render.overlay import Renderer, angle_keys_for
from .skeleton import h36m_to_coco
from .synth import SquatSpec, project, squat_set

SCRIPTS = {
    "none": [{}],
    "mixed": [{}, {"valgus": 8}, {}, {"chest_drop": 25}, {"depth": 85}, {"down_s": 0.4, "up_s": 0.5}],
    "valgus": [{"valgus": 8}],
    "chest": [{"chest_drop": 25}],
    "shallow": [{"depth": 85}],
}


def backdrop(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    grad = np.linspace(46, 22, h).astype(np.uint8)
    img[:] = grad[:, None, None]
    floor = int(h * 0.80)
    img[floor:] = (34, 38, 42)
    for x in range(-w, 2 * w, 80):
        cv2.line(img, (w // 2 + (x - w // 2) // 4, floor), (x, h), (48, 52, 56), 1, cv2.LINE_AA)
    cv2.line(img, (0, floor), (w, floor), (70, 74, 78), 2)
    return img


def run_demo(out: Path | None = None, faults: str = "mixed", reps: int = 6, fps: float = 30.0) -> Path:
    script = SCRIPTS[faults]
    per_rep = [script[i % len(script)] for i in range(reps)]
    seq, _ = squat_set(per_rep, SquatSpec(fps=fps, yaw=25.0))
    W, H = 1280, 720
    kp_h36 = project(seq, (W, H), dist_mm=4200, focal=1000)
    kp = h36m_to_coco(kp_h36)
    ex = get_exercise("squat")
    coach = Coach(ex, fps=fps, up_mode="camera")
    r = Renderer(ex.display, angle_keys_for(ex))
    out = out or sessions_dir() / "demo" / "demo_squat.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    bg = backdrop(W, H)
    writer = None
    for i, pose in enumerate(seq):
        t = i / fps
        u = coach.ingest(i, t, pose)
        if i == len(seq) - 1:
            u.reps += coach.finish()
        for rep in u.reps:
            r.note_rep(t, rep)
        for c in u.cues:
            r.note_cue(t, c)
        r.yaw = -60.0 + 25.0 * np.sin(t * 0.4)
        frame = r.render(bg.copy(), kp[i], u, t, status="synthetic athlete")
        if writer is None:
            writer = VideoWriter(out, fps, (frame.shape[1] - frame.shape[1] % 2, frame.shape[0] - frame.shape[0] % 2))
        writer.write(frame)
    if writer:
        writer.close()
    s = coach.summary()
    print(f"{s['reps']} reps, mean score {s['mean_score']}; faults: "
          + ", ".join(f"{f['fault']} x{f['reps']}" for f in s["faults"]))
    return out
