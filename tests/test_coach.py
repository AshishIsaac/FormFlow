"""Exercise definitions, the rule engine and the coach on synthetic athletes with known faults."""

import numpy as np
import pytest

from formflow.coach.exercises import Rule, exercises, get_exercise, parse_exercise
from formflow.coach.rules import evaluate_rep, live_violations
from formflow.pipeline import Coach, CueManager
from formflow.signal.reps import Rep
from formflow.synth import SquatSpec, squat_sequence, squat_set


def run(seq, ex="squat", fps=30.0):
    c = Coach(get_exercise(ex), fps=fps, up_mode="camera")
    cues = []
    for i, p in enumerate(seq):
        cues += c.ingest(i, i / fps, p).cues
    c.finish()
    return c, cues


def test_every_exercise_file_is_valid():
    ex = exercises()
    assert {"squat", "deadlift", "push_up", "lunge", "bicep_curl", "overhead_press", "pull_up", "jumping_jack",
            "situp", "bench_press", "front_raise", "auto"} <= set(ex)
    for e in ex.values():
        assert e.display and e.signal
        if not e.auto:
            assert e.rules, e.name
            assert e.reps.prior is not None, e.name
    assert get_exercise("Push-Up").name == "push_up"
    assert get_exercise("squant").name == "squat"  # RepCount's label spelling
    with pytest.raises(KeyError):
        get_exercise("underwater basket weaving")


def test_rule_validation():
    with pytest.raises(ValueError):
        Rule(id="x", fault="f", cue="c", metric="knee_flexion", agg="sideways", max=1)
    with pytest.raises(ValueError):
        Rule(id="x", fault="f", cue="c", metric="knee_flexion")  # no limit
    with pytest.raises(ValueError):
        parse_exercise({"name": "e", "signal": "knee_flexion", "rules": [
            {"id": "a", "fault": "f", "cue": "c", "metric": "m", "max": 1},
            {"id": "a", "fault": "f", "cue": "c", "metric": "m", "max": 1}]})


def test_clean_squats_score_100_and_reps_match_ground_truth():
    seq, bounds = squat_sequence(SquatSpec(reps=5))
    c, cues = run(seq)
    s = c.summary()
    assert s["reps"] == 5 and s["mean_score"] == 100.0 and not s["faults"]
    for r, (gs, gb, ge) in zip(s["rep_list"], bounds):
        assert abs(r["start_frame"] - gs) <= 2 and abs(r["bottom_frame"] - gb) <= 2 and abs(r["end_frame"] - ge) <= 2
    assert cues[:5] == ["1", "2", "3", "4", "5"]


@pytest.mark.parametrize("fault,kw,rule", [
    ("valgus", {"valgus": 8}, "knee_valgus"),
    ("chest", {"chest_drop": 25}, "chest_drop"),
    ("shallow", {"depth": 85}, "depth"),
    ("fast", {"down_s": 0.35, "up_s": 0.5}, "tempo"),
    ("shift", {"asym": 25}, "shift"),
    ("back", {"back_round": 30}, "rounded_back"),
])
def test_each_injected_fault_is_named_on_the_faulty_rep_only(fault, kw, rule):
    seq, _ = squat_set([{}, kw, {}])
    c, cues = run(seq)
    s = c.summary()
    assert s["reps"] == 3
    faulted = [r["index"] for r in s["rep_list"] if any(f["id"] == rule for f in r["faults"])]
    assert faulted == [2], (fault, [(r["index"], [f["id"] for f in r["faults"]]) for r in s["rep_list"]])
    assert s["rep_list"][0]["score"] == 100 and s["rep_list"][1]["score"] < 100
    assert s["rep_list"][1]["faults"][0]["cue"] in cues  # the rep's main fault is spoken


def test_valgus_reports_the_worse_side_and_live_highlights():
    seq, _ = squat_set([{}, {"valgus": 9}])
    c = Coach(get_exercise("squat"), fps=30.0, up_mode="camera")
    live_ids = set()
    for i, p in enumerate(seq):
        live_ids |= {v["id"] for v in c.ingest(i, i / 30, p).live}
    c.finish()
    f = next(f for f in c.reps[1].faults if f.id == "knee_valgus")
    assert f.side in ("left", "right") and f.severity == "major"
    assert "knee_valgus" in live_ids


def test_auto_mode_finds_the_rep_signal():
    seq, _ = squat_sequence(SquatSpec(reps=6, noise_mm=5))
    c, _ = run(seq, "auto")
    assert c.count == 6
    assert c.active in ("knee_flexion", "hip_flexion")


def test_rule_types_on_crafted_traces():
    ex = parse_exercise({"name": "t", "signal": "a", "reps": {"prior": [0, 100]}, "rules": [
        {"id": "lim", "fault": "F", "cue": "C", "metric": "a", "agg": "max", "min": 90, "scale": 10, "weight": 20},
        {"id": "tempo", "type": "tempo", "fault": "T", "cue": "C", "phase": "out", "min": 1.0, "scale": 0.5, "weight": 10},
        {"id": "ratio", "type": "ratio", "fault": "R", "cue": "C", "a": "a", "b": "b", "phase": "up", "frac": 0.5,
         "max": 1.5, "scale": 1.0, "weight": 10},
        {"id": "rise", "fault": "S", "cue": "C", "metric": "b", "agg": "rise", "max": 5, "scale": 5, "weight": 10, "live": True},
    ]})
    n = 61
    a = np.concatenate([np.linspace(0, 80, 21), np.linspace(80, 0, 40)])  # peaks at 80 (< 90)
    b = np.concatenate([np.linspace(0, 20, 21), np.linspace(20, 15, 40)])  # hip barely moves while "a" drops
    rep = Rep(index=1, start_frame=0, bottom_frame=20, end_frame=n - 1, t_start=0, t_bottom=0.5, t_end=2.0,
              peak_value=80, start_value=0, excursion=0.8)
    score, faults = evaluate_rep(ex, rep, {"a": a, "b": b})
    ids = {f.id for f in faults}
    assert ids == {"lim", "tempo", "ratio", "rise"}
    assert 0 <= score < 100
    lim = next(f for f in faults if f.id == "lim")
    assert lim.value == pytest.approx(80) and lim.limit == 90
    assert [r.id for r, _ in live_violations(ex, {"b": 30.0}, "down", {"b": 10.0})] == ["rise"]
    assert live_violations(ex, {"b": 30.0}, "top", {"b": 10.0}) == []


def test_cue_manager_does_not_nag():
    cm = CueManager(repeat_s=8.0)
    rep = Rep(1, 0, 5, 10, 0, 0.5, 1.0, 90, 0, 1.0)
    from formflow.coach.rules import Fault

    rep.faults = [Fault("v", "Knees in", "Knees out", 0.6, 0.75, "major", 20, 5)]
    assert cm.on_rep(rep, 1.0) == ["1", "Knees out"]
    rep.index = 2
    assert cm.on_rep(rep, 3.0) == ["2"]  # same cue within 8 s: only the count
    rep.index = 3
    assert cm.on_rep(rep, 10.0) == ["3", "Knees out"]


def test_front_view_rules_are_skipped_when_filmed_side_on():
    front, _ = squat_set([{"valgus": 9, "asym": 25}], SquatSpec(yaw=0))
    side, _ = squat_set([{"valgus": 9, "asym": 25}], SquatSpec(yaw=88))
    ids = lambda seq: {f["id"] for r in run(seq)[0].summary()["rep_list"] for f in r["faults"]}
    assert {"knee_valgus", "shift"} <= ids(front)
    assert not ({"knee_valgus", "shift"} & ids(side))  # one leg hidden behind the other: not judged
