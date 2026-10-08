"""Build ``samples/``: short openly licensed clips + FormFlow's annotated analysis of each.

    python scripts/fetch_samples.py COMMONS_DIR          # downloads the sources (Wikimedia Commons)
    python scripts/make_samples.py COMMONS_DIR [--only name] [--no-analyze]

For each entry in SAMPLES: trim + transcode to 720p H.264 without audio (``samples/<name>.mp4``), write
its provenance and licence (``samples/<name>.json``), run ``formflow analyze`` on it and keep the
annotated video (``samples/<name>_annotated.mp4``). ``samples/README.md`` lists every attribution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples"

SAMPLES = [
    {"name": "squat_barbell", "src": "Squat_-_exercise_demonstration_video.webm", "exercise": "squat",
     "title": "Barbell back squat", "start": 0, "dur": None},
    {"name": "pushup_side", "src": "Army_Combat_Fitness_Test-_Hand-Release_Push-Up_HRP_Event_3_.webm",
     "exercise": "push_up", "title": "Hand-release push-ups (side view, bystanders)", "start": 31, "dur": 24},
    {"name": "lunge_forward", "src": "Strength_Training_Circuit-_Forward_Lunge.webm", "exercise": "lunge",
     "title": "Forward lunge with kettlebells", "start": 27, "dur": 20},
    {"name": "push_press", "src": "Strength_Training_Circuit-_Overhead_Push_Press.webm", "exercise": "overhead_press",
     "title": "Kettlebell push press", "start": 30, "dur": 16},
    {"name": "deadlift_trapbar", "src": "Army_Combat_Fitness_Test-_3_Repetition_Maximum_Deadlift_MDL_Event_1_.webm",
     "exercise": "deadlift", "title": "Trap-bar deadlift (3 reps, crowded field)", "start": 41, "dur": 26},
    {"name": "deadlift_barbell", "src": "Deadlift_-_exercise_demonstration_video.webm", "exercise": "deadlift",
     "title": "Barbell deadlift (rear view)", "start": 0, "dur": None},
    {"name": "pullup", "src": "Pull-ups_-_exercise_demonstration_video.webm", "exercise": "pull_up",
     "title": "Pull-ups (rear view)", "start": 0, "dur": None},
    {"name": "shoulder_press", "src": "Shoulder_press_-_exercise_demonstration_video.webm", "exercise": "overhead_press",
     "title": "Barbell shoulder press (rear view)", "start": 0, "dur": None},
    {"name": "bench_press", "src": "Bench_press_-_exercise_demonstration_video.webm", "exercise": "bench_press",
     "title": "Bench press", "start": 0, "dur": None},
]


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def transcode(src: Path, dst: Path, start: float, dur: float | None) -> None:
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-ss", str(start), "-i", str(src)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-an", "-vf", "scale=-2:720,fps=30", "-c:v", "libx264", "-preset", "slow", "-crf", "27",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("commons", type=Path)
    ap.add_argument("--only")
    ap.add_argument("--no-analyze", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    for s in SAMPLES:
        if a.only and s["name"] != a.only:
            continue
        src = a.commons / s["src"]
        meta = json.loads((a.commons / (s["src"] + ".json")).read_text(encoding="utf-8"))
        vid = OUT / f"{s['name']}.mp4"
        print(f"{s['name']}: {s['src']} [{meta['license']}]", flush=True)
        transcode(src, vid, s["start"], s["dur"])
        info = {"title": s["title"], "exercise": s["exercise"], "video": vid.name,
                "source": meta["page"], "license": meta["license"], "license_url": meta.get("license_url", ""),
                "author": meta.get("author") or meta.get("credit", ""), "trim": {"start_s": s["start"], "duration_s": s["dur"]},
                "modifications": "trimmed, scaled to 720p, re-encoded (H.264), audio removed"}
        if not a.no_analyze:
            from formflow.analyze import analyze_video
            from formflow.config import load_config
            from formflow.paths import sessions_dir

            out_dir = sessions_dir() / f"sample_{s['name']}"
            r = analyze_video(vid, s["exercise"], load_config(), out_dir=out_dir)
            ann = OUT / f"{s['name']}_annotated.mp4"
            # re-encode the annotated video smaller for the repository
            subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(out_dir / "annotated.mp4"), "-an",
                            "-vf", "scale=-2:540", "-c:v", "libx264", "-preset", "slow", "-crf", "28",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(ann)], check=True)
            sm = r["summary"]
            info.update(annotated=ann.name, reps=sm["reps"], mean_score=sm["mean_score"],
                        faults=[f"{f['fault']} ({f['reps']}/{sm['reps']})" for f in sm["faults"]],
                        note=f"{sm['reps']} reps, score {sm['mean_score']}")
            print(f"   -> {sm['reps']} reps, score {sm['mean_score']}, faults: {info['faults']}", flush=True)
        (OUT / f"{s['name']}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    write_readme()
    return 0


def write_readme() -> None:
    rows = []
    for f in sorted(OUT.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        res = f"{m['reps']} reps, score {m['mean_score']}" if "reps" in m else ""
        rows.append(f"| `{m['video']}` | {m['title']} | {m['exercise']} | {res} | [{m['license']}]({m['license_url'] or m['source']}) "
                    f"| {m['author'] or 'see source'} | [Commons]({m['source']}) |")
    text = ("# Samples\n\nShort openly licensed exercise clips from Wikimedia Commons (US Army training videos in the public "
            "domain, and CC BY / CC0 clips), each with FormFlow's annotated analysis (`*_annotated.mp4`). The clips "
            "were trimmed, scaled to 720p, re-encoded and stripped of audio; FormFlow's overlays were added to the "
            "annotated copies. Rebuild with `python scripts/fetch_samples.py DIR && python scripts/make_samples.py DIR`.\n\n"
            "| file | content | exercise | FormFlow result | licence | author | source |\n|---|---|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n\n`demo_synthetic.mp4` is rendered by `formflow demo` (synthetic athlete, no camera).\n")
    (OUT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
