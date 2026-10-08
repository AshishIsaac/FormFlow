"""Joint angles, One Euro filter and the rep state machine."""

import numpy as np
import pytest

from formflow.signal.angles import (
    UpEstimator,
    angle_between,
    compute_metrics,
    flexion,
    knee_ankle_ratio,
    trunk_inclination,
)
from formflow.signal.one_euro import OneEuroFilter, filter_sequence
from formflow.signal.reps import RepCounter, RepParams, periodicity, segment
from formflow.skeleton import J, coco_to_h36m
from formflow.synth import SquatSpec, squat_sequence

# ---------------------------------------------------------------------------- angles


def test_flexion_of_a_right_angle_and_a_straight_limb():
    p = np.zeros((17, 3))
    p[J["r_hip"]] = [0, 0, 0]
    p[J["r_knee"]] = [0, 1, 0]
    p[J["r_ankle"]] = [0, 2, 0]
    assert flexion(p, J["r_hip"], J["r_knee"], J["r_ankle"]) == pytest.approx(0, abs=1e-6)
    p[J["r_ankle"]] = [0, 1, 1]
    assert flexion(p, J["r_hip"], J["r_knee"], J["r_ankle"]) == pytest.approx(90, abs=1e-6)
    assert angle_between(np.array([1, 0, 0]), np.array([0, 0, 2])) == pytest.approx(90)


@pytest.mark.parametrize("depth", [60.0, 95.0, 125.0])
def test_synthetic_squat_knee_flexion_is_recovered_exactly(depth):
    seq, bounds = squat_sequence(SquatSpec(reps=1, depth=depth))
    m = compute_metrics(seq)
    s, b, e = bounds[0]
    assert m["knee_flexion"][b] == pytest.approx(depth, abs=0.5)
    assert m["knee_flexion"][0] == pytest.approx(5.0, abs=0.5)
    assert m["knee_flexion_asym"][b] == pytest.approx(0, abs=0.5)


def test_metrics_do_not_depend_on_camera_yaw():
    a = compute_metrics(squat_sequence(SquatSpec(reps=1, yaw=0))[0])
    b = compute_metrics(squat_sequence(SquatSpec(reps=1, yaw=70))[0])
    for k in ("knee_flexion", "hip_flexion", "trunk_inclination", "spine_flexion", "knee_ankle_ratio", "knee_medial_max"):
        np.testing.assert_allclose(a[k], b[k], atol=1e-3, err_msg=k)


def test_valgus_measures_grow_with_knee_collapse():
    vals = []
    for v in (0, 4, 8):
        seq, bounds = squat_sequence(SquatSpec(reps=1, valgus=v))
        m = compute_metrics(seq)
        b = bounds[0][1]
        vals.append((m["knee_ankle_ratio"][b], m["knee_medial_max"][b], m["knee_valgus_max"][b]))
    kasr, medial, fppa = zip(*vals)
    assert kasr[0] == pytest.approx(1.0, abs=0.02) and kasr[0] > kasr[1] > kasr[2]
    assert medial[0] <= 0.5 < 3 < medial[2] and fppa[0] < fppa[2]


def test_trunk_inclination_and_up_estimation_correct_a_tilted_camera():
    seq, _ = squat_sequence(SquatSpec(reps=1))
    a = np.radians(20)  # camera rolled 20 degrees
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    tilted = seq @ R.T
    assert trunk_inclination(tilted[0]) > 15  # naive camera-up reading is wrong by ~the tilt
    up = UpEstimator(rate=0.2)
    for p in tilted[:30]:  # the athlete stands tall before the first rep
        up.update(p)
    assert trunk_inclination(tilted[0], up.up) < 6


def test_kasr_and_coco_conversion():
    p = np.zeros((17, 3))
    p[J["l_knee"]], p[J["r_knee"]] = [1, 0, 0], [-1, 0, 0]
    p[J["l_ankle"]], p[J["r_ankle"]] = [2, 1, 0], [-2, 1, 0]
    assert knee_ankle_ratio(p) == pytest.approx(0.5)
    coco = np.random.default_rng(0).normal(size=(5, 17, 3))
    h = coco_to_h36m(coco)
    np.testing.assert_allclose(h[:, J["pelvis"]], (coco[:, 11] + coco[:, 12]) / 2)
    np.testing.assert_allclose(h[:, J["l_knee"]], coco[:, 13])
    np.testing.assert_allclose(h[:, J["spine"]], (h[:, J["pelvis"]] + h[:, J["thorax"]]) / 2)


# ---------------------------------------------------------------------------- One Euro


def test_one_euro_removes_jitter_when_still_and_follows_fast_motion():
    rng = np.random.default_rng(1)
    fps = 30.0
    still = 100 + rng.normal(0, 2.0, (300, 1))
    f = filter_sequence(still, fps, min_cutoff=1.0, beta=0.05)
    assert np.std(f[60:]) < 0.5 * np.std(still[60:])
    t = np.arange(300) / fps
    ramp = (200 * t)[:, None] + rng.normal(0, 2.0, (300, 1))  # 200 px/s
    g = filter_sequence(ramp, fps, min_cutoff=1.0, beta=0.05)
    lag = np.mean((200 * t)[60:] - g[60:, 0])
    ema = np.empty_like(ramp)
    ema[0] = ramp[0]
    for i in range(1, len(ramp)):
        ema[i] = 0.2 * ramp[i] + 0.8 * ema[i - 1]  # an EMA with similar smoothing when still
    assert lag < 0.5 * np.mean((200 * t)[60:] - ema[60:, 0])


def test_one_euro_holds_low_confidence_points():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f(np.array([[0.0, 0.0]]), 0.0)
    out = f(np.array([[100.0, 100.0]]), 1 / 30, conf=np.array([0.05]))
    assert np.all(out < 5)  # a probably-wrong jump barely moves the filtered point
    out = f(np.array([[100.0, 100.0]]), 2 / 30, conf=np.array([1.0]))
    assert np.all(out > 12)  # a confident one moves it at the normal rate


# ---------------------------------------------------------------------------- reps


def _wave(n_reps=6, period=2.0, fps=30.0, amp=80.0, base=10.0, noise=0.0, lead=1.0, seed=0):
    t = np.arange(int((lead + n_reps * period + 1.0) * fps)) / fps
    x = np.where((t > lead) & (t < lead + n_reps * period), 0.5 - 0.5 * np.cos(2 * np.pi * (t - lead) / period), 0)
    return base + amp * x + np.random.default_rng(seed).normal(0, noise, len(t))


@pytest.mark.parametrize("noise", [0.0, 3.0])
@pytest.mark.parametrize("amp", [40.0, 110.0])
def test_counter_counts_clean_and_noisy_reps_of_any_range(noise, amp):
    reps, phases = segment(_wave(noise=noise, amp=amp), 30.0, RepParams(prior=(15, 100), min_rom=25))
    assert len(reps) == 6
    assert set(phases) >= {"top", "down", "up"}
    for r in reps:
        assert r.t_start < r.t_bottom < r.t_end
        assert 1.5 < r.duration_s < 2.6


def test_counter_ignores_fidgeting_and_flags_partials():
    rc = RepCounter(RepParams(prior=(15, 100), min_rom=25))
    sig = _wave(amp=8.0, noise=1.0)  # standing and shifting weight
    for i, v in enumerate(sig):
        rc.update(i, i / 30, v)
    assert rc.count == 0
    # deep reps, then one half rep, then deep again
    full = _wave(n_reps=3, amp=90.0, lead=0.5)
    half = _wave(n_reps=1, amp=35.0, lead=0.0)
    seq = np.concatenate([full, half, full])
    rc = RepCounter(RepParams(prior=(15, 100), min_rom=25))
    for i, v in enumerate(seq):
        rc.update(i, i / 30, v)
    rc.flush()
    assert rc.count == 6
    assert len(rc.partials) == 1


def test_counter_handles_decreasing_signals_and_bounces():
    sig = 200 - _wave(amp=80.0)  # e.g. a height that drops during the rep
    reps, _ = segment(sig, 30.0, RepParams(prior=(190, 110), min_rom=25))
    assert len(reps) == 6
    # a small bounce at the bottom must not split one rep into two
    t = np.arange(150) / 30
    one = 10 + 80 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 5))
    one[70:80] -= 10 * np.sin(np.linspace(0, np.pi, 10))
    reps, _ = segment(np.concatenate([np.full(30, 10.0), one, np.full(30, 10.0)]), 30.0, RepParams(prior=(15, 100)))
    assert len(reps) == 1


def test_periodicity_prefers_the_repetitive_signal():
    rng = np.random.default_rng(3)
    rep = _wave(amp=60.0, noise=1.0)
    junk = 30 + np.cumsum(rng.normal(0, 0.5, len(rep)))
    assert periodicity(rep, 30.0) > 3 * periodicity(junk, 30.0)


def test_a_rep_already_under_way_at_the_start_is_not_counted():
    full = _wave(n_reps=3, amp=90.0, lead=0.5)
    start = int(0.5 * 30 + 0.75 * 60)  # begin three quarters of the way through the first rep
    reps, _ = segment(full[start:], 30.0, RepParams(prior=(15, 100), min_rom=25))
    assert len(reps) == 2
