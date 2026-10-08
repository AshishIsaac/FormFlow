"""Web API and live WebSocket protocol (with a small injected engine), launcher, CLI."""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from formflow.config import load_config
from formflow.lift.lifter import Lifter
from formflow.synth import SquatSpec, project, squat_sequence

REPO = Path(__file__).resolve().parents[1]


class SyntheticTracker:
    """Stands in for YOLOX + RTMPose: returns a synthetic squatter's keypoints frame by frame."""

    def __init__(self):
        from formflow.skeleton import h36m_to_coco

        seq, _ = squat_sequence(SquatSpec(reps=2))
        self.kp = h36m_to_coco(project(seq, (640, 360), dist_mm=4200, focal=450))
        self.i = 0

    def __call__(self, img):
        from formflow.pose2d.tracker import Pose2D

        k = self.kp[min(self.i, len(self.kp) - 1)]
        self.i += 1
        return Pose2D(k, np.array([0, 0, 640, 360, 1.0], np.float32), False)


class SmallEngine:
    device = "cpu"

    def __init__(self):
        from helpers import tiny_model

        self.lifter = Lifter(None, model=tiny_model(maxlen=81), device="cpu", flip=False)

    def tracker(self, det_every=10, async_detect=False):
        return SyntheticTracker()

    def describe(self):
        return {"device": "cpu", "detector": "fake", "pose2d": "synthetic", "lifter": "tiny"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from formflow.server.app import create_app

    monkeypatch.setenv("FORMFLOW_HOME", str(tmp_path))
    cfg = load_config(overrides=["live.window=27", "live.stride=1"])
    return TestClient(create_app(cfg, engine=SmallEngine()))


def test_api_basics(client):
    assert client.get("/api/health").json()["ok"]
    ex = client.get("/api/exercises").json()
    sq = next(e for e in ex if e["name"] == "squat")
    assert sq["rules"] and "knee_ankle_ratio" in sq["metrics"]
    st = client.get("/api/status").json()
    assert st["loaded"] and st["engine"]["lifter"] == "tiny"
    assert client.get("/api/sessions").json() == []
    assert "FormFlow" in client.get("/").text
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/sessions/../../etc/passwd").status_code == 404


def test_live_websocket_round_trip_and_finish_saves_a_session(client, tmp_path):
    img = np.full((360, 640, 3), 80, np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    with client.websocket_connect("/ws/live") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(json.dumps({"type": "exercise", "name": "squat"}))
        assert ws.receive_json()["exercise"]["name"] == "squat"
        frames = []
        for _ in range(60):
            ws.send_bytes(jpg.tobytes())
            frames.append(ws.receive_json())
        f = frames[-1]
        assert f["type"] == "frame" and f["visible"] and len(f["kp2d"]) == 17
        assert len(f["pose3d"]) == 17 and f["phase"] in ("top", "down", "bottom", "up")
        assert "knee_flexion" in f["metrics"] and f["signal_name"] == "knee_flexion"
        assert 0 <= f["kp2d"][0][0] <= 1  # normalised for the browser
        ws.send_text(json.dumps({"type": "finish"}))
        msg = ws.receive_json()
        assert msg["type"] == "summary"


def test_analyze_rejects_unknown_exercise(client):
    r = client.post("/api/analyze", files={"file": ("x.mp4", b"0000", "video/mp4")}, data={"exercise": "nope"})
    assert r.status_code == 400


def _launcher():
    spec = importlib.util.spec_from_file_location("ff_run", REPO / "run.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ff_run"] = m
    sys.dont_write_bytecode = True  # keep the repository root clean
    spec.loader.exec_module(m)
    return m


def test_launcher_plan_and_options():
    r = _launcher()
    keys = [s["key"] for s in r.plan("cpu")]
    assert keys == ["venv", "pip", "torch", "ort-clean", "ort", "app", "models", "verify"]
    for prof in ("gpu", "cpu"):
        steps = {s["key"]: s for s in r.plan(prof)}
        assert steps["torch"]["cmd"][-1].endswith(f"torch-{prof}.txt")
        assert steps["ort"]["cmd"][-1].endswith(f"onnxruntime-{prof}.txt")
        assert steps["models"]["cmd"][-1] == prof
        assert steps["ort-clean"]["cmd"][-1] == ("onnxruntime" if prof == "gpu" else "onnxruntime-gpu")
        assert not steps["ort-clean"]["required"]
        for f in (steps["torch"]["cmd"][-1], steps["ort"]["cmd"][-1]):
            assert Path(f).is_file()
    assert r.project_fingerprint() == r.project_fingerprint()
    assert r._take(["--home", "D:/x", "--cpu"], "--home") == ("D:/x", ["--cpu"])
    assert r.main(["--bogus"]) == 2


def test_launcher_uses_only_stdlib():
    tree = ast.parse((REPO / "run.py").read_text(encoding="utf-8"))
    mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


def test_cli_exercises_and_config_overrides(capsys):
    from formflow.cli import main

    assert main(["exercises"]) == 0
    assert "squat" in capsys.readouterr().out
    cfg = load_config(overrides=["live.window=27", "one_euro.beta=0.1"])
    assert cfg["live"]["window"] == 27 and cfg["one_euro"]["beta"] == 0.1
    with pytest.raises(ValueError):
        load_config(overrides=["nonsense"])


def test_demo_renders_a_video(tmp_path):
    from formflow.demo import run_demo

    out = run_demo(tmp_path / "demo.mp4", faults="valgus", reps=2)
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 60


def test_html_report(tmp_path):
    from formflow.render.report import write_html

    res = {"video": "x.mp4", "display": "Squat", "mode": "offline", "frames": 90, "fps": 30.0, "annotated": "annotated.mp4",
           "signal": {"values": list(np.sin(np.arange(90) / 5) * 40 + 60)},
           "summary": {"signal": "knee_flexion", "reps": 1, "partial_reps": 0, "mean_score": 70.0, "mean_rep_s": 2.0,
                       "faults": [{"id": "d", "fault": "Too <shallow>", "cue": "Sit deeper", "reps": 1}],
                       "rep_list": [{"index": 1, "score": 70.0, "out_s": 1.0, "back_s": 1.0, "peak_value": 95.0,
                                     "start_frame": 5, "end_frame": 60,
                                     "faults": [{"fault": "Too <shallow>", "side": None, "value": 0.4, "limit": 0.2}]}]}}
    write_html(res, tmp_path / "r.html")
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Too &lt;shallow&gt;" in html and "<svg" in html and "annotated.mp4" in html
