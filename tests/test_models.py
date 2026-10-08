"""The real models (marker ``models``): MotionBERT on projected synthetic squats, 2D + analysis on samples."""

from pathlib import Path

import numpy as np
import pytest

from formflow.data.download import have_model, model_path

pytestmark = pytest.mark.models
REPO = Path(__file__).resolve().parents[1]


def _need(*names):
    missing = [n for n in names if not have_model(n)]
    if missing:
        pytest.skip(f"models not downloaded: {missing} (formflow models download)")


@pytest.mark.parametrize("variant", ["lite", "full"])
def test_motionbert_recovers_synthetic_squat_angles_and_reps(variant):
    _need(f"motionbert-{variant}")
    from formflow.coach.exercises import get_exercise
    from formflow.lift.lifter import Lifter
    from formflow.pipeline import Coach
    from formflow.signal.angles import compute_metrics
    from formflow.synth import SquatSpec, project, squat_sequence

    seq, bounds = squat_sequence(SquatSpec(reps=4, yaw=30))
    kp = project(seq)
    lf = Lifter(str(model_path(f"motionbert-{variant}")), variant, device="cpu", flip=False)
    pred = lf.lift_sequence(kp, (1280, 720), window=81, overlap=0.5)
    gt_k = compute_metrics(seq)["knee_flexion"]
    pr_k = compute_metrics(pred)["knee_flexion"]
    r = np.corrcoef(gt_k, pr_k)[0, 1]
    assert r > 0.95, r
    assert np.mean(np.abs(gt_k - pr_k)) < 15
    c = Coach(get_exercise("squat"), fps=30.0, up_mode="camera")
    for i, p in enumerate(pred):
        c.ingest(i, i / 30, p)
    c.finish()
    assert c.count == 4


def _sample(name):
    p = REPO / "samples" / f"{name}.mp4"
    if not p.is_file():
        pytest.skip(f"sample {p.name} missing")
    return p


def test_2d_models_find_the_athlete_in_a_sample():
    _need("yolox-tiny", "rtmpose-m")
    import cv2

    from formflow.pose2d.models import YOLOX, RTMPose
    from formflow.pose2d.tracker import PoseTracker

    cap = cv2.VideoCapture(str(_sample("squat_barbell")))
    tr = PoseTracker(YOLOX(str(model_path("yolox-tiny")), "cpu"), RTMPose(str(model_path("rtmpose-m")), "cpu"))
    confs = []
    for _ in range(30):
        ok, img = cap.read()
        if not ok:
            break
        confs.append(tr(img).keypoints[:, 2].mean())
    assert np.mean(confs) > 0.5


def test_analyze_a_sample_end_to_end(tmp_path):
    _need("yolox-tiny", "rtmpose-m", "motionbert-lite")
    from formflow.analyze import analyze_video
    from formflow.config import load_config

    cfg = load_config(overrides=["device=cpu", "offline.flip=false"])
    r = analyze_video(_sample("squat_barbell"), "squat", cfg, out_dir=tmp_path, render=True)
    assert r["summary"]["reps"] >= 2
    assert (tmp_path / "annotated.mp4").stat().st_size > 10_000
    assert (tmp_path / "report.html").is_file()
