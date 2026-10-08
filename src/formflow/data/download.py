"""Model and dataset downloads (resumable, retried, verified), all into FORMFLOW_HOME.

Models
  yolox-m / yolox-tiny   person detectors (OpenMMLab RTMPose ONNX SDK, Apache-2.0)
  rtmpose-m              17-keypoint body model trained on 7 public datasets (Apache-2.0)
  motionbert-full/-lite  MotionBERT 3D-pose checkpoints fine-tuned on Human3.6M (Apache-2.0)

Datasets (benchmarks only - the app needs none of them)
  h36m      Human3.6M S9/S11 test set as used by MotionBERT: stacked-hourglass 2D detections +
            Vicon 3D. Only the test arrays are kept (~340 MB). Human3.6M's licence allows research
            use only; register at http://vision.imar.ro/human3.6m if you use it.
  repcount  RepCount-A test split (TransRAC, CVPR 2022): 152 videos with repetition counts and
            cycle boundaries, read from a public Lance export on Hugging Face.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from ..paths import data_dir, models_dir

OMM = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
HF_MB = "https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/"

MODELS: dict[str, dict] = {
    "yolox-m": {"url": OMM + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip", "member": "end2end.onnx",
                "file": "yolox-m.onnx", "bytes": 94223081, "about": "person detector, 640 px (GPU default)"},
    "yolox-tiny": {"url": OMM + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", "member": "end2end.onnx",
                   "file": "yolox-tiny.onnx", "bytes": 18887737, "about": "person detector, 416 px (CPU default)"},
    "rtmpose-m": {"url": OMM + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
                  "member": "end2end.onnx", "file": "rtmpose-m.onnx", "bytes": 50799818,
                  "about": "RTMPose-m body keypoints, 256x192"},
    "motionbert-full": {"url": HF_MB + "FT_MB_release_MB_ft_h36m/best_epoch.bin", "file": "motionbert-full.bin",
                        "bytes": 169958969, "about": "MotionBERT 3D lifter, 42M params (optional: --set lifter=motionbert-full)"},
    "motionbert-lite": {"url": HF_MB + "FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
                        "file": "motionbert-lite.bin", "bytes": 64099897,
                        "about": "MotionBERT-Lite 3D lifter, 16M params (default)"},
}

PROFILES = {"gpu": ["yolox-m", "rtmpose-m", "motionbert-lite"],
            "cpu": ["yolox-tiny", "rtmpose-m", "motionbert-lite"]}

H36M_URL = ("https://huggingface.co/datasets/CameronSteele/h36m_3dhp/resolve/main/"
            "h36m_sh_conf_cam_source_final.pkl.zip")
REPCOUNT_URI = "hf://datasets/lmms-lab-eval/repcounta-lance/data/test.lance"


def _say(msg: str) -> None:
    print(msg, flush=True)


def fetch(url: str, dest: Path, expected_bytes: int | None = None, retries: int = 6) -> Path:
    """Download with resume (HTTP Range) and retries. Writes ``dest.part`` then renames."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        return dest
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.is_file() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "FormFlow/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else expected_bytes
                if have and r.status != 206:  # server ignored the range: start over
                    have = 0
                    part.unlink(missing_ok=True)
                t0, last = time.time(), 0.0
                with part.open("ab" if have else "wb") as f:
                    done = have
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        now = time.time()
                        if now - last > 2 and total:
                            last = now
                            rate = (done - have) / max(now - t0, 1e-3) / 1e6
                            _say(f"  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB ({rate:.1f} MB/s)")
            if expected_bytes and part.stat().st_size != expected_bytes:
                raise OSError(f"size mismatch: got {part.stat().st_size}, expected {expected_bytes}")
            part.replace(dest)
            return dest
        except Exception as e:  # network hiccup: back off and resume
            last_err = e
            _say(f"  download interrupted ({e}); retry {attempt}/{retries} ...")
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"could not download {url}: {last_err}")


def model_path(name: str) -> Path:
    return models_dir() / MODELS[name]["file"]


def have_model(name: str) -> bool:
    return model_path(name).is_file()


def download_model(name: str) -> Path:
    spec = MODELS[name]
    out = model_path(name)
    if out.is_file():
        return out
    _say(f"Downloading {name} - {spec['about']} ({spec['bytes'] / 1e6:.0f} MB)")
    if "member" in spec:
        z = fetch(spec["url"], models_dir() / (name + ".zip"), spec["bytes"])
        with zipfile.ZipFile(z) as zf:
            member = next(m for m in zf.namelist() if m.endswith(spec["member"]))
            with zf.open(member) as src, out.with_suffix(".tmp").open("wb") as dst:
                shutil.copyfileobj(src, dst)
        out.with_suffix(".tmp").replace(out)
        z.unlink()
    else:
        fetch(spec["url"], out, spec["bytes"])
    return out


def download_models(names: list[str]) -> list[Path]:
    return [download_model(n) for n in names]


# --------------------------------------------------------------------------- datasets


def h36m_path() -> Path:
    return data_dir() / "h36m" / "h36m_test.npz"


def download_h36m() -> Path:
    """MotionBERT's H36M pickle -> just the test arrays as a compressed-free npz (needs ~1.8 GB while converting)."""
    import pickle

    import numpy as np

    out = h36m_path()
    if out.is_file():
        return out
    d = out.parent
    _say("Downloading the Human3.6M benchmark arrays (674 MB, research use only - see the README)")
    z = fetch(H36M_URL, d / "h36m.pkl.zip")
    with zipfile.ZipFile(z) as zf:
        member = next(m for m in zf.namelist() if m.endswith(".pkl") and "__MACOSX" not in m)
        _say("  extracting ...")
        with zf.open(member) as f:
            test = pickle.load(f)["test"]
    np.savez(out.with_suffix(".tmp.npz"), joint_2d=test["joint_2d"], confidence=test["confidence"][..., 0],
             joints_25d=test["joints_2.5d_image"], factor=test["2.5d_factor"], camera=test["camera_name"],
             action=np.array(test["action"]), source=np.array(test["source"]))
    out.with_suffix(".tmp.npz").replace(out)
    z.unlink()
    return out


def repcount_dir() -> Path:
    return data_dir() / "repcount"


def download_repcount(limit: int | None = None) -> Path:
    """RepCount-A test videos + annotations (needs the ``pylance`` package: pip install -e .[bench])."""
    try:
        import lance
    except ImportError as e:
        raise SystemExit("The RepCount benchmark reads a Lance dataset: pip install -e .[bench]") from e
    root = repcount_dir()
    vids = root / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    ann_file = root / "annotations.json"
    ds = lance.dataset(REPCOUNT_URI)
    meta = ds.to_table(columns=["video_id", "action_type", "count", "cycle_bounds_json", "video_ext",
                                "video_size_bytes"]).to_pylist()
    if limit:
        meta = meta[:limit]
    def ext(m: dict) -> str:
        e = (m.get("video_ext") or "mp4").strip()
        return e if e.startswith(".") else "." + e

    ann = []
    todo = [i for i, m in enumerate(meta) if not (vids / f"{m['video_id']}{ext(m)}").is_file()]
    total = sum(meta[i]["video_size_bytes"] for i in todo)
    _say(f"RepCount-A test split: {len(meta)} videos, {len(todo)} to fetch ({total / 1e6:.0f} MB)")
    for k in range(0, len(todo), 8):
        rows = todo[k:k + 8]
        blobs = ds.take_blobs("video_blob", indices=rows)
        for i, b in zip(rows, blobs):
            m = meta[i]
            p = vids / f"{m['video_id']}{ext(m)}"
            with b as f:
                data = f.read()
            p.with_suffix(".part").write_bytes(data)
            p.with_suffix(".part").replace(p)
        _say(f"  {min(k + 8, len(todo))}/{len(todo)} videos")
    for m in meta:
        ann.append({"video": f"videos/{m['video_id']}{ext(m)}", "id": m["video_id"],
                    "action": m["action_type"], "count": m["count"],
                    "cycles": json.loads(m["cycle_bounds_json"] or "[]")})
    ann_file.write_text(json.dumps(ann, indent=1), encoding="utf-8")
    return ann_file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":  # python -m formflow.data.download <model...>
    download_models(sys.argv[1:] or PROFILES["gpu"])
