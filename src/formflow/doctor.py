"""``formflow doctor``: check the installation end to end and say what to fix."""

from __future__ import annotations

import platform
import sys
import time


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", flush=True)


def doctor(cfg: dict, quick: bool = False) -> int:
    from . import __version__

    print(f"FormFlow {__version__} - Python {platform.python_version()} on {platform.system()}")
    bad = 0
    for mod in ("numpy", "cv2", "yaml", "fastapi", "uvicorn", "onnxruntime", "torch"):
        try:
            m = __import__(mod)
            _ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            _fail(f"{mod}: {e}")
            bad += 1
    if bad:
        return 1

    import numpy as np
    import onnxruntime as ort
    import torch

    if torch.cuda.is_available():
        _ok(f"CUDA GPU: {torch.cuda.get_device_name(0)} (PyTorch {torch.version.cuda})")
    else:
        _warn("no CUDA GPU: running on the CPU (live coaching at a lower frame rate)")
    provs = ort.get_available_providers()
    if torch.cuda.is_available() and "CUDAExecutionProvider" not in provs:
        _warn(f"onnxruntime has no CUDA provider ({provs}): 2D pose runs on the CPU")

    # The coach on a synthetic athlete: angles, reps, rules, cues - no models needed.
    from .coach.exercises import exercises, get_exercise
    from .pipeline import Coach
    from .synth import SquatSpec, squat_set

    n_ex = len(exercises())
    _ok(f"{n_ex} exercise definitions load")
    seq, bounds = squat_set([{}, {"valgus": 8}, {}], SquatSpec())
    coach = Coach(get_exercise("squat"), fps=30.0, up_mode="camera")
    for i, p in enumerate(seq):
        coach.ingest(i, i / 30.0, p)
    coach.finish()
    s = coach.summary()
    faulted = [r["index"] for r in s["rep_list"] if any(f["id"] == "knee_valgus" for f in r["faults"])]
    if s["reps"] == 3 and faulted == [2]:
        _ok("coach: 3/3 synthetic reps counted, knee cave-in caught on rep 2 only")
    else:
        _fail(f"coach: expected 3 reps with a fault on rep 2, got {s['reps']} reps, faults on {faulted}")
        bad += 1
    if quick:
        return 1 if bad else 0

    # Models: present, loadable, and a forward pass on real-shaped input.
    from .data.download import have_model
    from .engine import build_engine, pick_models

    names = pick_models(cfg)
    missing = [n for k, n in names.items() if k != "device" and not have_model(n)]
    if missing:
        _warn(f"downloading missing models: {', '.join(missing)}")
    try:
        t0 = time.time()
        eng = build_engine(cfg, "live")
        _ok(f"models loaded in {time.time() - t0:.1f}s: {eng.describe()}")
    except Exception as e:
        _fail(f"models: {e}")
        return 1
    from .lift.lifter import StreamingLifter
    from .synth import project

    kp = project(seq[:60])
    from .config import live_settings

    lv = live_settings(cfg, eng.lifter.device)
    stream = StreamingLifter(eng.lifter, (1280, 720), window=int(lv["window"]), stride=int(lv["stride"]))
    t0 = time.perf_counter()
    outs = [o for i, k in enumerate(kp) for o in stream.push(i, k, force=(i == len(kp) - 1))]
    dt = (time.perf_counter() - t0) / len(kp) * 1000
    pose = outs[-1][1]
    if np.isfinite(pose).all() and float(np.ptp(pose[:, 1])) > 0:
        _ok(f"3D lifter runs ({dt:.1f} ms per frame live: window {lv['window']}, stride {lv['stride']}, {eng.lifter.device})")
    else:
        _fail("3D lifter produced invalid output")
        bad += 1
    img = np.full((480, 640, 3), 90, np.uint8)
    t0 = time.perf_counter()
    eng.detector(img)
    eng.pose(img, np.array([[200, 50, 440, 470, 1.0]], np.float32))
    _ok(f"2D models run ({(time.perf_counter() - t0) * 1000:.0f} ms incl. warm-up, {eng.pose.device})")
    print("All good." if not bad else f"{bad} problem(s).")
    return 1 if bad else 0


if __name__ == "__main__":
    from .config import load_config

    sys.exit(doctor(load_config()))
