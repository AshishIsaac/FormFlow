"""Biomechanics rule engine: turns one rep's joint-angle traces into a score and named faults.

Rule types (see ``configs/exercises/*.yaml``):

limit  aggregate a metric over a window of the rep (min, max, p90, value at the start / bottom / end,
       range, rise above the starting posture ...) and compare it with ``min`` / ``max``.
tempo  duration of the ``out`` half (start -> far end), the ``back`` half, or the ``total`` rep.
ratio  how much metric ``a`` changes relative to metric ``b`` over the first ``frac`` of a window
       (deadlift: knees straightening much faster than the hips = "hips shooting up").

A violated rule costs ``weight * (0.4 + 0.6 * min(1, excess / scale))`` points from 100, so a fault
that is just over the line is a small deduction and one ``scale`` beyond it costs the full weight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..signal.reps import Rep
from .exercises import Exercise, Rule

MAJOR = 0.5  # excess / scale at which a fault is reported as major
SIDE_ON = 0.75  # mean side_view above which `view: front` rules are skipped (one side of the body is hidden)


@dataclass
class Fault:
    id: str
    fault: str
    cue: str
    value: float
    limit: float
    severity: str  # "minor" | "major"
    penalty: float
    frame: int  # where it was worst (absolute frame index)
    side: str | None = None
    joints: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = round(self.value, 3)
        d["penalty"] = round(self.penalty, 1)
        d["joints"] = list(self.joints)
        return d


def _window(rule_windows: list[str], rep: Rep, n: int, offset: int, excursion: np.ndarray | None) -> np.ndarray:
    """Frame indices (relative to the metric arrays) of the requested phases of this rep."""
    s, b, e = rep.start_frame - offset, rep.bottom_frame - offset, rep.end_frame - offset
    s, b, e = max(s, 0), min(max(b, 0), n - 1), min(max(e, 0), n - 1)
    idx: list[np.ndarray] = []
    for w in rule_windows:
        if w == "rep":
            idx.append(np.arange(s, e + 1))
        elif w == "down":
            idx.append(np.arange(s, b + 1))
        elif w == "up":
            idx.append(np.arange(b, e + 1))
        elif w == "top":
            idx.append(np.array([s, e]))
        elif w == "bottom":
            if excursion is not None:
                rng = np.arange(s, e + 1)
                sel = rng[excursion[rng] >= 0.8 * rep.excursion]
                idx.append(sel if len(sel) else np.array([b]))
            else:
                idx.append(np.arange(max(s, b - 2), min(e, b + 2) + 1))
    out = np.unique(np.concatenate(idx)) if idx else np.arange(s, e + 1)
    return out[(out >= 0) & (out < n)]


def _aggregate(x: np.ndarray, agg: str, w: np.ndarray, rep: Rep, offset: int, n: int) -> tuple[float, int]:
    """Aggregate metric ``x`` over window ``w``. Returns (value, frame of the deciding sample)."""
    xs = x[w]
    good = np.isfinite(xs)
    if not good.any():
        return float("nan"), int(w[0]) if len(w) else 0
    xs, w = xs[good], w[good]
    s = min(max(rep.start_frame - offset, 0), n - 1)
    e = min(max(rep.end_frame - offset, 0), n - 1)
    b = min(max(rep.bottom_frame - offset, 0), n - 1)
    if agg == "min":
        i = int(np.argmin(xs))
        return float(xs[i]), int(w[i])
    if agg == "max":
        i = int(np.argmax(xs))
        return float(xs[i]), int(w[i])
    if agg in ("mean", "median"):
        v = float(np.mean(xs) if agg == "mean" else np.median(xs))
        return v, int(w[int(np.argmin(np.abs(xs - v)))])
    if agg in ("p10", "p90"):
        v = float(np.percentile(xs, 10 if agg == "p10" else 90))
        return v, int(w[int(np.argmin(np.abs(xs - v)))])
    if agg in ("start", "end", "bottom"):
        # A few frames around the event, so one noisy frame doesn't decide.
        c = {"start": s, "end": e, "bottom": b}[agg]
        lo, hi = max(c - 2, 0), min(c + 2, n - 1)
        seg = x[lo:hi + 1]
        seg = seg[np.isfinite(seg)]
        return (float(np.median(seg)) if len(seg) else float("nan")), c
    if agg == "range":
        return float(xs.max() - xs.min()), int(w[int(np.argmax(np.abs(xs - np.median(xs))))])
    if agg == "rise":
        base = x[max(s - 2, 0):s + 3]
        base = base[np.isfinite(base)]
        b0 = float(np.median(base)) if len(base) else float(xs[0])
        i = int(np.argmax(xs))
        return float(xs[i] - b0), int(w[i])
    raise ValueError(agg)


def _penalty(rule: Rule, value: float) -> tuple[float, float, float] | None:
    """(penalty, limit, excess) when the value breaks the rule, else None."""
    if not np.isfinite(value):
        return None
    if rule.max is not None and value > rule.max:
        limit, excess = rule.max, value - rule.max
    elif rule.min is not None and value < rule.min:
        limit, excess = rule.min, rule.min - value
    else:
        return None
    frac = min(1.0, excess / rule.scale)
    return rule.weight * (0.4 + 0.6 * frac), limit, excess


def evaluate_rep(ex: Exercise, rep: Rep, metrics: dict[str, np.ndarray], offset: int = 0,
                 excursion: np.ndarray | None = None) -> tuple[float, list[Fault]]:
    """Score one rep. ``metrics[name][i]`` is frame ``offset + i``; ``excursion`` the counter's 0..1 depth."""
    n = len(next(iter(metrics.values())))
    faults: list[Fault] = []
    side_on = False
    if "side_view" in metrics:
        w_all = _window(["rep"], rep, n, offset, None)
        sv = metrics["side_view"][w_all] if len(w_all) else metrics["side_view"]
        side_on = bool(np.nanmean(sv) > SIDE_ON)
    for rule in ex.rules:
        if rule.view == "front" and side_on:
            continue
        side = None
        frame = rep.bottom_frame
        if rule.type == "tempo":
            value = {"out": rep.out_s, "back": rep.back_s, "total": rep.duration_s}[str(rule.phase)]
        elif rule.type == "ratio":
            w = _window(rule.windows, rep, n, offset, excursion)
            k = max(2, int(round(len(w) * rule.frac)))
            if len(w) < 3 or rule.a not in metrics or rule.b not in metrics:
                continue
            a, b = metrics[rule.a][w[:k]], metrics[rule.b][w[:k]]
            da, db = abs(float(a[-1] - a[0])), abs(float(b[-1] - b[0]))
            value = da / max(db, 3.0)  # 3 degree floor: a still hip is not an infinite ratio
            frame = int(w[k - 1]) + offset
        else:
            if rule.metric not in metrics:
                continue
            w = _window(rule.windows, rep, n, offset, excursion)
            if not len(w):
                continue
            value, fi = _aggregate(metrics[rule.metric], rule.agg, w, rep, offset, n)
            frame = fi + offset
            if rule.sides and all(s_ in metrics for s_ in rule.sides):
                # ``sides`` are per-side metrics where larger = worse (e.g. knee_medial_l / _r).
                vals = [float(np.nanmax(metrics[s_][w])) for s_ in rule.sides]
                side = "left" if rule.sides[int(np.argmax(vals))].endswith("_l") else "right"
        hit = _penalty(rule, value)
        if hit is None:
            continue
        pen, limit, excess = hit
        joints = tuple(rule.joints)
        if side and len(joints) == 2:
            joints = tuple(j for j in joints if j.startswith(side[0] + "_")) or joints
        faults.append(Fault(id=rule.id, fault=rule.fault, cue=rule.cue, value=value, limit=limit,
                            severity="major" if excess / rule.scale >= MAJOR else "minor", penalty=pen,
                            frame=frame, side=side, joints=joints))
    faults.sort(key=lambda f: -f.penalty)
    score = max(0.0, 100.0 - sum(f.penalty for f in faults))
    return round(score, 1), faults


def live_violations(ex: Exercise, frame_metrics: dict[str, float], phase: str,
                    baseline: dict[str, float] | None = None) -> list[tuple[Rule, float]]:
    """Instantaneous checks for ``live: true`` rules while a rep is in progress.

    Per-frame values stand in for the rep aggregate (min / max / percentiles); ``rise`` rules compare
    with ``baseline``, the metrics at the start of the current rep. ``range`` / ``start`` / ``end``
    rules only make sense over a whole rep and are not checked live.
    """
    out = []
    side_on = frame_metrics.get("side_view", 0.0) > SIDE_ON
    for rule in ex.rules:
        if not rule.live or rule.type != "limit" or rule.agg in ("range", "start", "end"):
            continue
        if rule.view == "front" and side_on:
            continue
        if phase == "top":
            continue
        wins = rule.windows
        if "rep" not in wins and phase not in wins:
            continue
        v = frame_metrics.get(rule.metric or "")
        if v is None:
            continue
        if rule.agg == "rise":
            if not baseline or rule.metric not in baseline:
                continue
            v = v - baseline[rule.metric]
        hit = _penalty(rule, float(v))
        if hit is not None:
            out.append((rule, float(v)))
    return out
