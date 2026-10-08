"""``formflow bench ...``: run the benchmarks and write ``BENCHMARK.md``.

    formflow bench lift  [--quick]          joint-angle / MPJPE error vs Human3.6M mocap
    formflow bench reps  [--quick] [--mode offline|live|both]   rep counting on RepCount-A
    formflow bench speed                    live per-frame latency on this machine
    formflow bench report                   rebuild BENCHMARK.md from the JSON results
    formflow publish                        copy the results into benchmarks/results (committed)

Results go to FORMFLOW_HOME/bench/results; ``publish`` copies them next to the code.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from ..paths import REFERENCE_DIR, bench_dir


def results_dir() -> Path:
    d = bench_dir() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _machine() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor()}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- 3D / joint angles


def lift_configs(cfg: dict, quick: bool):
    from .lift_eval import LiftConfig

    oe2 = cfg.get("one_euro")
    oe3 = cfg.get("live", {}).get("one_euro_3d")
    cams = ["54138969"]  # live rows: camera 1 of 4, the first 20 sequences (the causal path is ~10x slower)
    mv = 4 if quick else 20
    off = 8 if quick else None
    live = dict(causal=True, stride=2, flip=False, half=True, norm="crop", cameras=cams, max_videos=mv)
    return [
        # The published MotionBERT evaluation (image normalisation, 243-frame clips, flip): checks the port.
        LiftConfig("MotionBERT protocol (offline, full)", "full", 243, norm="image", flip=True, max_videos=off),
        # What `formflow analyze` does (crop normalisation, overlapping clips, One Euro on 2D).
        # (no 2D One Euro offline: with future frames the filter only adds lag - see analyze.smooth_2d)
        LiftConfig("FormFlow offline - full", "full", 243, overlap=0.5, norm="crop", flip=True, max_videos=off),
        LiftConfig("FormFlow offline - lite", "lite", 243, overlap=0.5, norm="crop", flip=True, max_videos=off),
        LiftConfig("FormFlow offline - full, with 2D One Euro", "full", 243, overlap=0.5, norm="crop", flip=True,
                   one_euro=oe2, max_videos=off),
        # Live: the shipped default and its ablations.
        LiftConfig("live default - lite, 81 frames, 3D One Euro", "lite", 81, one_euro_3d=oe3, **live),
        LiftConfig("live - lite, 81 frames, 2D+3D One Euro", "lite", 81, one_euro=oe2, one_euro_3d=oe3, **live),
        LiftConfig("live - full, 81 frames, 2D+3D One Euro", "full", 81, one_euro=oe2, one_euro_3d=oe3, **live),
        LiftConfig("live - lite, 27 frames, 2D+3D One Euro", "lite", 27, one_euro=oe2, one_euro_3d=oe3, **live),
        LiftConfig("live - lite, 81 frames, no filters", "lite", 81, **live),
        LiftConfig("live - lite, 81 frames, 2D One Euro only", "lite", 81, one_euro=oe2, **live),
        # Framing robustness: the athlete at half the H36M size in the frame.
        LiftConfig("zoom 0.5x - image normalisation", "lite", 243, norm="image", flip=False, zoom=0.5, max_videos=mv),
        LiftConfig("zoom 0.5x - crop normalisation", "lite", 243, norm="crop", flip=False, zoom=0.5, max_videos=mv),
    ]


def bench_lift(cfg: dict, quick: bool = False, only: str | None = None) -> list[dict]:
    """``only``: run just the matching configurations (see ``wanted``) and merge them into the saved results."""
    from ..data.download import download_h36m, download_model, model_path
    from .lift_eval import evaluate, load_h36m_test

    data = load_h36m_test(download_h36m())
    path = results_dir() / "lift_h36m.json"
    prev = {r["config"]["name"]: r for r in json.loads(path.read_text(encoding="utf-8"))["results"]}         if (only and path.is_file()) else {}
    configs = lift_configs(cfg, quick)
    pats = [p.strip() for p in (only or "").split("|") if p.strip()]

    def wanted(name: str) -> bool:  # "|"-separated; "=name" matches exactly, anything else as a substring
        return not pats or any(name == p[1:] if p.startswith("=") else p in name for p in pats)

    out = []
    for c in configs:
        if not wanted(c.name):
            if c.name in prev:
                out.append(prev[c.name])
            continue
        name = f"motionbert-{c.variant}"
        download_model(name)
        print(f"[lift] {c.name}", flush=True)
        t0 = time.time()
        r = evaluate(c, data, str(model_path(name)))
        r["wall_s"] = round(time.time() - t0, 1)
        print(f"   MPJPE {r['mpjpe_mm']} mm  PA {r['pa_mpjpe_mm']} mm  angle MAE {r['angle_mae_deg']} deg  "
              f"accel {r['accel_mm']}  ({r['frames']} frames, {r['wall_s']}s)", flush=True)
        out.append(r)
    payload = {"quick": quick, "machine": _machine(), "date": time.strftime("%Y-%m-%d"), "results": out}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    write_report()
    return out


# --------------------------------------------------------------------------- rep counting


def bench_reps(cfg: dict, quick: bool = False, mode: str = "offline", limit: int | None = None) -> dict:
    from ..data.download import download_repcount
    from ..engine import build_engine
    from .reps_eval import run

    download_repcount(limit)
    modes = ["offline", "live"] if mode == "both" else [mode]
    res = {}
    for m in modes:
        engine = build_engine(cfg, m)  # live mode uses the live lifter (live.lifter)
        print(f"[reps] RepCount-A test, {m} lifting", flush=True)
        r = run(engine, cfg, bench_dir() / "repcount_cache", mode=m, limit=limit or (30 if quick else None))
        res[m] = r
        a = r["all"]
        print(f"   all {a['exercise']['n']} videos: MAE {a['exercise']['mae']}  OBO {a['exercise']['obo']}  "
              f"| auto: MAE {a['auto']['mae']}  OBO {a['auto']['obo']}", flush=True)
    path = results_dir() / "reps_repcount.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    old.update({"quick": quick, "machine": _machine(), "date": time.strftime("%Y-%m-%d")})
    old.setdefault("modes", {}).update(res)
    path.write_text(json.dumps(old, indent=1), encoding="utf-8")
    write_report()
    return res


# --------------------------------------------------------------------------- speed


def bench_speed(cfg: dict, video: str | None = None, seconds: float = 20.0) -> dict:
    """Live per-frame latency: the real LivePipeline fed as fast as it can go."""
    from ..analyze import iter_frames
    from ..coach.exercises import get_exercise
    from ..data.download import repcount_dir
    from ..engine import build_engine
    from ..paths import SAMPLES_DIR
    from ..pipeline import LivePipeline

    if video is None:
        cands = sorted(SAMPLES_DIR.glob("*.mp4")) + sorted((repcount_dir() / "videos").glob("*.mp4"))
        cands = [c for c in cands if "annotated" not in c.name]
        if not cands:
            raise FileNotFoundError("no video to time: pass one, or download RepCount")
        video = str(cands[0])
    out = {"video": Path(video).name, "machine": _machine(), "date": time.strftime("%Y-%m-%d"), "runs": []}
    devices = ["cuda", "cpu"]
    for dev in devices:
        if dev == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    continue
            except Exception:
                continue
        c = json.loads(json.dumps(cfg))
        c["device"] = dev
        eng = build_engine(c, "live")
        from ..config import live_settings

        lv = live_settings(c, eng.lifter.device)
        eng.lifter.flip = False
        eng.lifter.half = bool(lv.get("half", True)) and eng.lifter.device == "cuda"
        pipe = None
        tim = {"pose2d_ms": [], "lift_ms": [], "coach_ms": [], "total_ms": []}
        t_start = time.perf_counter()
        for i, img in enumerate(iter_frames(video, int(lv.get("max_side", 960)))):
            if pipe is None:
                pipe = LivePipeline(eng.tracker(int(lv.get("det_every", 10)), async_detect=True), eng.lifter,
                                    get_exercise("squat"), (img.shape[1], img.shape[0]), window=int(lv["window"]),
                                    stride=int(lv["stride"]), one_euro=c.get("one_euro"),
                                    one_euro_3d=lv.get("one_euro_3d"), filter_2d_for_lift=bool(lv.get("one_euro_2d_lift", False)))
            t0 = time.perf_counter()
            res = pipe.process(img, i / 30.0)
            dt = 1000 * (time.perf_counter() - t0)
            if i >= 30:  # warm-up excluded
                for k in ("pose2d_ms", "lift_ms", "coach_ms"):
                    tim[k].append(res.timings.get(k, 0.0))
                tim["total_ms"].append(dt)
            if time.perf_counter() - t_start > seconds:
                break
        stats = {k: {"mean": round(float(np.mean(v)), 2), "p50": round(float(np.median(v)), 2),
                     "p95": round(float(np.percentile(v, 95)), 2)} for k, v in tim.items() if v}
        run = {"device": dev, "models": eng.describe(), "frames": len(tim["total_ms"]),
               "fps": round(1000 / stats["total_ms"]["mean"], 1), "window": lv["window"], "stride": lv["stride"], **stats}
        print(f"[speed] {dev}: {run['fps']} fps  total p50 {stats['total_ms']['p50']} ms  p95 {stats['total_ms']['p95']} ms  "
              f"(2D {stats['pose2d_ms']['p50']} / 3D {stats['lift_ms']['p50']} ms)", flush=True)
        out["runs"].append(run)
    (results_dir() / "speed.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_report()
    return out


# --------------------------------------------------------------------------- report


def _load(name: str) -> dict | None:
    for d in (results_dir(), REFERENCE_DIR):
        p = d / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


ANGLE_LABELS = {"knee_flexion_l": "Knee flexion L", "knee_flexion_r": "Knee flexion R", "hip_flexion_l": "Hip flexion L",
                "hip_flexion_r": "Hip flexion R", "elbow_flexion_l": "Elbow flexion L",
                "elbow_flexion_r": "Elbow flexion R", "arm_elevation_l": "Arm elevation L",
                "arm_elevation_r": "Arm elevation R", "trunk_inclination": "Trunk inclination",
                "spine_flexion": "Spine flexion", "knee_valgus_l": "Knee valgus (FPPA) L",
                "knee_valgus_r": "Knee valgus (FPPA) R"}


def write_report() -> Path:
    lines = ["# FormFlow benchmarks", "",
             "Generated by `formflow bench report` from the JSON files in this folder. Re-run with "
             "`formflow bench lift`, `formflow bench reps --mode both` and `formflow bench speed`.", ""]
    lift = _load("lift_h36m.json")
    if lift:
        res = lift["results"]
        m = lift.get("machine", {})
        lines += ["## 1. Joint-angle error against motion capture (Human3.6M)", "",
                  f"Human3.6M test subjects S9 + S11, input = stacked-hourglass 2D detections with confidences "
                  f"(the standard benchmark input), ground truth = Vicon mocap skeleton. "
                  f"{'Quick subset. ' if lift.get('quick') else ''}Machine: {m.get('gpu', m.get('cpu', '?'))}, {lift.get('date', '')}.",
                  "", "| configuration | frames | MPJPE (mm) | PA-MPJPE (mm) | joint-angle MAE (deg) | accel err (mm/f^2) | lifter ms/frame |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in res:
            lines.append(f"| {r['config']['name']} | {r['frames']:,} | {r['mpjpe_mm']:.1f} | {r['pa_mpjpe_mm']:.1f} | "
                         f"{r['angle_mae_deg']:.2f} | {r['accel_mm']:.2f} | {r['ms_per_frame']:.2f} |")
        lines += ["", "Zoom rows: the 2D input scaled by 0.5 about the image centre (an athlete half as tall in the "
                  "frame), first 20 sequences of all cameras. MPJPE in mm is meaningless there (the 2.5D depth "
                  "factor assumes the original scale); PA-MPJPE and the angles are scale-free. Live rows: camera "
                  "54138969, first 20 sequences, causal lifting exactly as in the app.", ""]
        best = next((r for r in res if r["config"]["name"] == "FormFlow offline - lite"), res[0])
        live = next((r for r in res if r["config"]["name"].startswith("live default")), None)
        lines += ["", "Per-angle error (mean / median / 90th percentile, degrees):", "",
                  "| angle | " + best["config"]["name"] + (f" | {live['config']['name']}" if live else "") + " |",
                  "|---|---:|" + ("---:|" if live else "")]
        for k, lab in ANGLE_LABELS.items():
            a = best["angles"].get(k)
            if not a:
                continue
            row = f"| {lab} | {a['mae']:.1f} / {a['median']:.1f} / {a['p90']:.1f} |"
            if live and k in live["angles"]:
                b = live["angles"][k]
                row += f" {b['mae']:.1f} / {b['median']:.1f} / {b['p90']:.1f} |"
            lines.append(row)
        lines += ["", "The frontal-plane knee projection angle (FPPA) is the least reliable measure from one camera "
                  "(it is ill-conditioned in deep flexion), which is why the knee-valgus rules use the knee-to-ankle "
                  "separation ratio at the bottom (squat) and the knee's medial offset (lunge) instead.", ""]
    reps = _load("reps_repcount.json")
    if reps:
        lines += ["## 2. Rep counting (RepCount-A test split)", "",
                  "152 real-world videos (TransRAC, CVPR 2022). FormFlow is **not trained on RepCount**: the "
                  "counter is the live coach's state machine on 3D joint angles. MAE = mean(|pred - gt| / gt), "
                  "OBO = share of videos off by at most one rep.", "",
                  "| lifting | videos | label -> exercise: MAE | OBO | exact | auto (no label): MAE | OBO |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for mode, r in reps.get("modes", {}).items():
            for subset, title in (("all", "all classes"), ("supported", "classes with a FormFlow profile")):
                e, a = r[subset]["exercise"], r[subset]["auto"]
                if not e.get("n"):
                    continue
                lines.append(f"| {mode} - {title} | {e['n']} | {e['mae']:.3f} | {e['obo']:.3f} | {e['exact']:.3f} | "
                             f"{a['mae']:.3f} | {a['obo']:.3f} |")
        lines += ["", "For reference, learned counters trained on RepCount-A's training split report (test split): "
                  "TransRAC MAE 0.443 / OBO 0.291 (Hu et al., CVPR 2022); PoseRAC MAE 0.236 / OBO 0.560 "
                  "(Yao et al., 2023).", ""]
        first = next(iter(reps.get("modes", {}).values()), None)
        if first:
            lines += ["Per class (label -> exercise setting, " + first["mode"] + " lifting):", "",
                      "| RepCount class | FormFlow profile | videos | MAE | OBO |", "|---|---|---:|---:|---:|"]
            for k, v in first["per_class"].items():
                e = v["exercise"]
                lines.append(f"| {k} | {v['formflow_exercise']} | {e['n']} | {e['mae']:.3f} | {e['obo']:.3f} |")
            lines.append("")
    sp = _load("speed.json")
    if sp:
        lines += ["## 3. Live speed", "", f"Video `{sp['video']}`, live pipeline (async re-detection, causal lifter), "
                  f"{sp.get('date', '')}.", "",
                  "| device | models | fps | total p50 / p95 (ms) | 2D p50 (ms) | 3D p50 (ms) |", "|---|---|---:|---:|---:|---:|"]
        for r in sp["runs"]:
            md = r["models"]
            lines.append(f"| {r['device']} ({sp['machine'].get('gpu', 'CPU') if r['device'] == 'cuda' else 'CPU'}) | "
                         f"{md.get('detector')}, {md.get('pose2d')}, {md.get('lifter')} (window {r['window']}, stride {r['stride']}) | "
                         f"{r['fps']} | {r['total_ms']['p50']} / {r['total_ms']['p95']} | {r['pose2d_ms']['p50']} | {r['lift_ms']['p50']} |")
        lines.append("")
    out = results_dir() / "BENCHMARK.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def publish() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for f in results_dir().glob("*"):
        if f.suffix in (".json", ".md"):
            shutil.copy2(f, REFERENCE_DIR / f.name)
            print(f"published {REFERENCE_DIR / f.name}")
