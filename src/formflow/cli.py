"""Command line: ``formflow <command>`` (or ``python -m formflow <command>``)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__


def _cfg(args):
    from .config import load_config

    return load_config(getattr(args, "config", None), getattr(args, "set", None) or [])


def cmd_serve(args) -> int:
    from .server.app import serve

    cfg = _cfg(args)
    return serve(cfg, host=args.host, port=args.port, open_browser=not args.no_browser, https=args.https)


def cmd_live(args) -> int:
    from .desktop import run_desktop

    return run_desktop(_cfg(args), source=args.video if args.video else args.camera, exercise=args.exercise,
                       speak=not args.mute, record=args.record, mirror=not args.no_mirror and not args.video)


def cmd_analyze(args) -> int:
    from .analyze import analyze_video
    from .paths import sessions_dir

    cfg = _cfg(args)
    out = Path(args.out) if args.out else sessions_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(args.video).stem}"
    last = [0.0, ""]

    def progress(stage: str, frac: float) -> None:
        if stage != last[1] or frac - last[0] >= 0.1 or frac >= 1:
            print(f"  {stage:<14} {frac * 100:5.1f}%", flush=True)
            last[:] = [frac, stage]

    r = analyze_video(args.video, args.exercise, cfg, out_dir=out, render=not args.no_render,
                      mode="live" if args.live else "offline", progress=progress)
    s = r["summary"]
    print(f"\n{s['display']}: {s['reps']} reps, mean form score {s['mean_score']}, {s['partial_reps']} partial")
    for f in s["faults"]:
        print(f"  - {f['fault']} in {f['reps']}/{s['reps']} reps -> {f['cue']}")
    print(f"\nReport: {out / 'report.html'}")
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    out = run_demo(Path(args.out) if args.out else None, faults=args.faults, reps=args.reps)
    print(f"Demo video: {out}")
    return 0


def cmd_exercises(args) -> int:
    from .coach.exercises import exercises

    for ex in exercises().values():
        print(f"{ex.name:<15} {ex.display:<22} signal={ex.signal:<18} rules: {', '.join(r.id for r in ex.rules) or '-'}")
        print(f"{'':<15} camera: {ex.camera}")
    return 0


def cmd_models(args) -> int:
    from .data.download import MODELS, PROFILES, download_models, have_model, model_path

    if args.action == "list":
        for k, v in MODELS.items():
            print(f"{'[x]' if have_model(k) else '[ ]'} {k:<16} {v['bytes'] / 1e6:6.0f} MB  {v['about']}")
            if have_model(k):
                print(f"    {model_path(k)}")
        return 0
    names = sorted(set(PROFILES["gpu"] + PROFILES["cpu"])) if args.profile == "all" else PROFILES[args.profile]
    if args.profile == "auto":
        from .engine import resolve_device

        names = PROFILES["gpu" if resolve_device("auto") == "cuda" else "cpu"]
    download_models(names)
    return 0


def cmd_data(args) -> int:
    from .data.download import download_h36m, download_repcount

    if args.dataset in ("h36m", "all"):
        print(download_h36m())
    if args.dataset in ("repcount", "all"):
        print(download_repcount(args.limit))
    return 0


def cmd_bench(args) -> int:
    from .bench import runner

    cfg = _cfg(args)
    if args.what == "lift":
        runner.bench_lift(cfg, quick=args.quick, only=args.only)
    elif args.what == "reps":
        runner.bench_reps(cfg, quick=args.quick, mode=args.mode, limit=args.limit)
    elif args.what == "speed":
        runner.bench_speed(cfg)
    elif args.what == "report":
        runner.write_report()
    return 0


def cmd_publish(args) -> int:
    from .bench.runner import publish

    publish()
    return 0


def cmd_doctor(args) -> int:
    from .doctor import doctor

    return doctor(_cfg(args), quick=args.quick)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="formflow", description="Markerless 3D pose and biomechanics coach.")
    p.add_argument("--version", action="version", version=f"formflow {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="extra YAML merged over configs/default.yaml")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a setting, e.g. live.window=27")

    s = sub.add_parser("serve", help="web app: live camera coaching, video analysis, sessions")
    common(s)
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--https", action="store_true", help="self-signed HTTPS so phones on the Wi-Fi can use their camera")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("live", help="desktop window: coach from a webcam (or a video played as a camera)")
    common(s)
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--video", help="play a video file as if it were the camera")
    s.add_argument("-e", "--exercise", default="squat")
    s.add_argument("--mute", action="store_true", help="no spoken cues")
    s.add_argument("--record", help="save the annotated view to this .mp4")
    s.add_argument("--no-mirror", action="store_true")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("analyze", help="analyse a recorded video: reps, scores, faults, annotated video, report")
    common(s)
    s.add_argument("video")
    s.add_argument("-e", "--exercise", default="auto")
    s.add_argument("-o", "--out")
    s.add_argument("--live", action="store_true", help="use the causal live pipeline instead of whole-clip lifting")
    s.add_argument("--no-render", action="store_true")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("demo", help="render a synthetic athlete doing squats (no camera or models needed)")
    s.add_argument("-o", "--out")
    s.add_argument("--reps", type=int, default=6)
    s.add_argument("--faults", default="mixed", choices=["none", "mixed", "valgus", "chest", "shallow"])
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("exercises", help="list the exercises and their form rules")
    s.set_defaults(fn=cmd_exercises)

    s = sub.add_parser("models", help="list / download the pose and lifting models")
    s.add_argument("action", choices=["list", "download"])
    s.add_argument("--profile", default="auto", choices=["auto", "gpu", "cpu", "all"])
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("data", help="download a benchmark dataset")
    s.add_argument("action", choices=["download"])
    s.add_argument("dataset", choices=["h36m", "repcount", "all"])
    s.add_argument("--limit", type=int, help="repcount: only the first N videos")
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("bench", help="benchmarks: joint-angle error vs mocap, rep counting, speed")
    common(s)
    s.add_argument("what", choices=["lift", "reps", "speed", "report"])
    s.add_argument("--quick", action="store_true", help="a subset - minutes instead of hours")
    s.add_argument("--mode", default="offline", choices=["offline", "live", "both"])
    s.add_argument("--limit", type=int)
    s.add_argument("--only", help="lift: re-run only the configurations whose name contains this, keep the rest")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("publish", help="copy benchmark results into benchmarks/results (committed)")
    s.set_defaults(fn=cmd_publish)

    s = sub.add_parser("doctor", help="check the installation end to end")
    common(s)
    s.add_argument("--quick", action="store_true", help="skip the model checks")
    s.set_defaults(fn=cmd_doctor)

    args = p.parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

