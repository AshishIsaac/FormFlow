"""Rep-count accuracy on RepCount-A (TransRAC, CVPR 2022), test split: 152 real-world videos.

Every video runs through the real pipeline (YOLOX + RTMPose -> One Euro -> MotionBERT -> joint angles
-> rep state machine). FormFlow is not trained on RepCount at all: the counter is the same hand-built
state machine the live coach uses. Two settings:

  exercise   the video's label is mapped to the FormFlow exercise (squat -> squat, ...), so the counter
             uses that exercise's rep signal; classes FormFlow has no profile for use ``auto``
  auto       every video counted with the exercise-agnostic ``auto`` profile (no label used)

Metrics as in the RepCount literature: MAE = mean(|pred - gt| / gt), OBO = share of videos whose count
is off by at most one. Expensive stages are cached per video (2D tracks, 3D poses) under
FORMFLOW_HOME/bench/repcount, so re-running with a different counter setting takes seconds.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..analyze import coach_sequence, lift, smooth_2d, track_2d
from ..coach.exercises import get_exercise
from ..data.download import repcount_dir

LABEL_TO_EXERCISE = {
    "squat": "squat", "squant": "squat", "push_up": "push_up", "pushups": "push_up", "pull_up": "pull_up",
    "pullups": "pull_up", "jump_jack": "jumping_jack", "jumpjacks": "jumping_jack", "situp": "situp",
    "bench_pressing": "bench_press", "benchpressing": "bench_press", "front_raise": "front_raise",
    "frontraise": "front_raise", "pommelhorse": "auto", "others": "auto",
}
CANON = {"squant": "squat", "pushups": "push_up", "pullups": "pull_up", "jumpjacks": "jump_jack",
         "benchpressing": "bench_pressing", "frontraise": "front_raise"}


def load_annotations(root: Path | None = None) -> list[dict]:
    root = root or repcount_dir()
    f = root / "annotations.json"
    if not f.is_file():
        raise FileNotFoundError("RepCount is not downloaded: formflow data download repcount")
    return json.loads(f.read_text(encoding="utf-8"))


def metrics(rows: list[dict], key: str) -> dict:
    if not rows:
        return {"n": 0}
    gt = np.array([r["gt"] for r in rows], float)
    pr = np.array([r[key] for r in rows], float)
    ok = gt > 0
    err = np.abs(pr - gt)
    return {"n": int(len(rows)), "mae": round(float(np.mean(err[ok] / gt[ok])), 4), "obo": round(float(np.mean(err <= 1)), 4),
            "exact": round(float(np.mean(err == 0)), 4), "mean_abs_err": round(float(err.mean()), 2)}


def run(engine, cfg: dict, cache: Path, mode: str = "offline", limit: int | None = None,
        log=print) -> dict:
    ann = load_annotations()
    if limit:
        ann = ann[:limit]
    root = repcount_dir()
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, a in enumerate(ann):
        vid = root / a["video"]
        if not vid.is_file():
            continue
        c2d = cache / f"{a['id']}_2d.npz"
        if c2d.is_file():
            z = np.load(c2d)
            kp, wh, fps = z["kp"], tuple(int(v) for v in z["wh"]), float(z["fps"])
        else:
            kp, _, wh, fps = track_2d(engine, vid, cfg, max_side=int(cfg.get("live", {}).get("max_side", 960)))
            np.savez_compressed(c2d, kp=kp, wh=np.array(wh), fps=fps)
        c3d = cache / f"{a['id']}_3d_{mode}.npz"
        kp_s = smooth_2d(kp, fps, cfg, mode)
        if c3d.is_file():
            poses = np.load(c3d)["poses"]
        else:
            poses = lift(engine, kp_s, wh, cfg, mode, fps)
            np.savez_compressed(c3d, poses=poses)
        visible = kp[:, :, 2].mean(1) > 0.2
        label = CANON.get(a["action"], a["action"])
        ex_name = LABEL_TO_EXERCISE.get(a["action"], "auto")
        coach_ex, _ = coach_sequence(get_exercise(ex_name), poses, fps, visible, cfg.get("up", "auto"))
        coach_auto, _ = coach_sequence(get_exercise("auto"), poses, fps, visible, cfg.get("up", "auto"))
        rows.append({"id": a["id"], "action": label, "exercise": ex_name, "gt": int(a["count"]),
                     "pred_exercise": coach_ex.count, "pred_auto": coach_auto.count, "auto_signal": coach_auto.active,
                     "frames": int(len(kp)), "fps": fps, "visible": round(float(visible.mean()), 3)})
        if log and (i + 1) % 10 == 0:
            m = metrics(rows, "pred_exercise")
            log(f"  [{mode}] {i + 1}/{len(ann)} videos  MAE {m['mae']:.3f}  OBO {m['obo']:.3f}  ({time.time() - t0:.0f}s)")
    per = defaultdict(list)
    for r in rows:
        per[r["action"]].append(r)
    supported = [r for r in rows if r["exercise"] != "auto"]
    return {
        "mode": mode, "videos": len(rows),
        "all": {"exercise": metrics(rows, "pred_exercise"), "auto": metrics(rows, "pred_auto")},
        "supported": {"exercise": metrics(supported, "pred_exercise"), "auto": metrics(supported, "pred_auto")},
        "per_class": {k: {"exercise": metrics(v, "pred_exercise"), "auto": metrics(v, "pred_auto"),
                          "formflow_exercise": v[0]["exercise"]} for k, v in sorted(per.items())},
        "rows": rows,
    }
