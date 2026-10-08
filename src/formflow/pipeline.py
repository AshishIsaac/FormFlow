"""The FormFlow pipeline: camera frame -> 2D keypoints -> 3D skeleton -> joint angles -> reps -> feedback.

    PoseTracker (YOLOX + RTMPose)        2D COCO keypoints with confidences, every frame
    OneEuroFilter                        jitter removed, confidence-aware
    StreamingLifter (MotionBERT)         3D H36M skeleton from the last ``window`` frames (causal)
    compute_metrics + UpEstimator        knee / hip / elbow angles, valgus, trunk, spine, ...
    RepCounter                           phases (top/down/bottom/up) and rep boundaries
    evaluate_rep + live_violations       per-rep score and named faults, live joint highlights
    CueManager                           what to say, without nagging

``Coach`` is everything after the 2D stage, so the same logic runs on live 2D poses, on offline 3D
(``analyze``: whole-video, non-causal lifting) and in tests (synthetic skeletons).
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .coach.exercises import Exercise
from .coach.rules import evaluate_rep, live_violations
from .lift.lifter import Lifter, StreamingLifter
from .signal.angles import CAMERA_UP, UpEstimator, compute_metrics
from .signal.one_euro import OneEuroFilter
from .signal.reps import Rep, RepCounter, periodicity
from .skeleton import coco_to_h36m

HISTORY_S = 90.0  # metrics kept for rule evaluation and plots


@dataclass
class CoachUpdate:
    """What changed on one step. ``pose3d`` / ``metrics`` belong to ``frame3d`` (may lag the camera)."""

    frame3d: int | None = None
    pose3d: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    signal: float | None = None
    phase: str = "top"
    count: int = 0
    excursion: float = 0.0
    reps: list[Rep] = field(default_factory=list)  # completed (counted) on this step, scored
    partials: list[Rep] = field(default_factory=list)
    live: list[dict] = field(default_factory=list)  # live rule violations: {id, fault, joints, value}
    cues: list[str] = field(default_factory=list)
    signal_name: str = ""
    up: np.ndarray | None = None  # estimated gravity direction (camera frame), for levelling the 3D view


class CueManager:
    """Turns reps and faults into short spoken / on-screen cues without repeating itself."""

    def __init__(self, repeat_s: float = 8.0, live_hold_s: float = 1.0) -> None:
        self.repeat_s = repeat_s
        self.live_hold_s = live_hold_s
        self._said: dict[str, float] = {}
        self._live_since: dict[str, float] = {}

    def _ok(self, key: str, t: float) -> bool:
        return t - self._said.get(key, -1e9) >= self.repeat_s

    def on_rep(self, rep: Rep, t: float) -> list[str]:
        out = [str(rep.index)]
        major = [f for f in rep.faults if f.severity == "major"] or rep.faults[:1]
        if major and self._ok(major[0].id, t):
            out.append(major[0].cue)
            self._said[major[0].id] = t
        elif not rep.faults and rep.index % 5 == 0:
            out.append("Good form, keep going")
        return out

    def on_live(self, live: list[dict], t: float) -> list[str]:
        active = {v["id"] for v in live}
        for k in list(self._live_since):
            if k not in active:
                del self._live_since[k]
        out = []
        for v in live:
            since = self._live_since.setdefault(v["id"], t)
            if t - since >= self.live_hold_s and self._ok(v["id"], t):
                out.append(v["cue"])
                self._said[v["id"]] = t
        return out


class Coach:
    def __init__(self, exercise: Exercise, fps: float = 30.0, up_mode: str = "auto") -> None:
        self.ex = exercise
        self.fps = fps
        self.up_mode = up_mode
        self.up_est = UpEstimator() if (up_mode == "auto" and exercise.posture == "standing") else None
        self.cues = CueManager()
        if exercise.auto:
            names = exercise.candidates or ["knee_flexion", "hip_flexion", "elbow_flexion", "arm_elevation"]
            self.counters = {n: RepCounter(exercise.reps) for n in names}
        else:
            self.counters = {exercise.signal: RepCounter(exercise.reps)}
        self.active = next(iter(self.counters))
        self._last_select = -1e9
        maxlen = int(HISTORY_S * fps) + 1
        self.h_frame: deque[int] = deque(maxlen=maxlen)
        self.h_t: deque[float] = deque(maxlen=maxlen)
        self.h_metrics: deque[dict[str, float]] = deque(maxlen=maxlen)
        self.h_exc: deque[float] = deque(maxlen=maxlen)
        self.reps: list[Rep] = []
        self.partials: list[Rep] = []
        self._rep_baseline: dict[str, float] = {}

    # ---- signal selection (auto mode) ----------------------------------------
    def _select(self, t: float) -> None:
        if len(self.counters) == 1 or t - self._last_select < 1.0 or len(self.h_metrics) < self.fps * 2:
            return
        self._last_select = t
        n = int(min(len(self.h_metrics), 20 * self.fps))
        recent = list(self.h_metrics)[-n:]
        best, best_s = self.active, -1.0
        for name in self.counters:
            sig = np.array([m[name] for m in recent])
            s = periodicity(sig, self.fps)
            if self.counters[name].count > 0:
                s *= 1.15  # mild stickiness toward a signal that is already counting
            if s > best_s:
                best, best_s = name, s
        self.active = best

    @property
    def count(self) -> int:
        return self.counters[self.active].count

    # ---- one 3D pose ----------------------------------------------------------
    def ingest(self, frame: int, t: float, pose3d: np.ndarray) -> CoachUpdate:
        up = self.up_est.update(pose3d) if self.up_est is not None else CAMERA_UP
        m = {k: float(v) for k, v in compute_metrics(pose3d, up).items()}
        self.h_frame.append(frame)
        self.h_t.append(t)
        self.h_metrics.append(m)
        upd = CoachUpdate(frame3d=frame, pose3d=pose3d, metrics=m, up=np.asarray(up))
        done: dict[str, Rep | None] = {}
        for name, rc in self.counters.items():
            prev_state = rc.state
            done[name] = rc.update(frame, t, m[name])
            if name == self.active and prev_state == "top" and rc.state != "top":
                self._rep_baseline = dict(self.h_metrics[-1])
        self._select(t)
        rc = self.counters[self.active]
        self.h_exc.append(rc.excursion)
        upd.signal, upd.signal_name = m[self.active], self.active
        upd.phase, upd.count, upd.excursion = rc.state, rc.count, rc.excursion
        rep = done.get(self.active)
        if rep is not None:
            if rep.counted:
                self._score(rep)
                self.reps.append(rep)
                upd.reps.append(rep)
                upd.cues += self.cues.on_rep(rep, t)
            else:
                self.partials.append(rep)
                upd.partials.append(rep)
        if not self.ex.auto:
            live = []
            base = self._rep_baseline if rc.state != "top" else {}
            for rule, v in live_violations(self.ex, m, rc.state, base):
                live.append({"id": rule.id, "fault": rule.fault, "cue": rule.cue, "joints": rule.joints,
                             "metric": rule.metric, "value": round(v, 2)})
            upd.live = live
            upd.cues += self.cues.on_live(live, t)
        return upd

    def finish(self) -> list[Rep]:
        """End of the session / video: complete a rep that is still waiting for its lockout."""
        out = []
        for name, rc in self.counters.items():
            rep = rc.flush()
            if rep is not None and rep.counted and name == self.active:
                self._score(rep)
                self.reps.append(rep)
                out.append(rep)
        return out

    def _score(self, rep: Rep) -> None:
        frames = np.array(self.h_frame)
        lo = int(np.searchsorted(frames, rep.start_frame - 3))
        hi = int(np.searchsorted(frames, rep.end_frame + 3, side="right"))
        hist = list(self.h_metrics)[lo:hi]
        if not hist:
            rep.score, rep.faults = 100.0, []
            return
        metrics = {k: np.array([h[k] for h in hist]) for k in hist[0]}
        exc = np.array(list(self.h_exc)[lo:hi] + [0.0] * max(0, len(hist) - len(list(self.h_exc)[lo:hi])))
        score, faults = evaluate_rep(self.ex, rep, metrics, offset=int(frames[lo]), excursion=exc)
        rep.score, rep.faults = score, faults

    # ---- summary ----------------------------------------------------------------
    def summary(self) -> dict:
        faults = Counter(f.id for r in self.reps for f in r.faults)
        names = {f.id: f.fault for r in self.reps for f in r.faults}
        cues = {f.id: f.cue for r in self.reps for f in r.faults}
        scores = [r.score for r in self.reps if r.score is not None]
        tempo = [r.duration_s for r in self.reps]
        return {
            "exercise": self.ex.name,
            "display": self.ex.display,
            "signal": self.active,
            "reps": len(self.reps),
            "partial_reps": len(self.partials),
            "mean_score": round(float(np.mean(scores)), 1) if scores else None,
            "best_rep": max(self.reps, key=lambda r: r.score or 0).index if self.reps else None,
            "mean_rep_s": round(float(np.mean(tempo)), 2) if tempo else None,
            "faults": [{"id": k, "fault": names[k], "cue": cues[k], "reps": v, "share": round(v / len(self.reps), 2)}
                       for k, v in faults.most_common()],
            "rep_list": [r.to_dict() | {"faults": [f.to_dict() for f in r.faults]} for r in self.reps],
            "up_vector": (self.up_est.up.round(3).tolist() if self.up_est is not None else CAMERA_UP.tolist()),
        }

    def counts_by_signal(self) -> dict[str, int]:
        return {k: rc.count for k, rc in self.counters.items()}


@dataclass
class FrameResult:
    frame: int
    t: float
    kp2d: np.ndarray  # (17, 3) COCO, filtered
    box: np.ndarray | None
    visible: bool
    update: CoachUpdate
    timings: dict[str, float]


class LivePipeline:
    """Frames in, coaching out. Owns the 2D tracker, the One Euro filter, the streaming lifter and a Coach."""

    def __init__(self, tracker, lifter: Lifter, exercise: Exercise, img_wh: tuple[int, int], fps: float = 30.0,
                 window: int = 81, stride: int = 2, one_euro: dict | None = None, one_euro_3d: dict | None = None,
                 up_mode: str = "auto", filter_2d_for_lift: bool = False) -> None:
        self.filter_2d_for_lift = filter_2d_for_lift
        self.tracker = tracker
        self.stream = StreamingLifter(lifter, img_wh, window=window, stride=stride)
        oe = {"min_cutoff": 1.5, "beta": 0.1} | (one_euro or {})
        self.filter = OneEuroFilter(**oe)
        self.filter3d = OneEuroFilter(**one_euro_3d) if one_euro_3d else None
        self.coach = Coach(exercise, fps=fps, up_mode=up_mode)
        self.img_wh = img_wh
        self.frame = 0
        self._last: CoachUpdate = CoachUpdate()

    def set_exercise(self, exercise: Exercise) -> None:
        self.coach = Coach(exercise, fps=self.coach.fps, up_mode=self.coach.up_mode)

    def process(self, img_bgr: np.ndarray, t: float | None = None) -> FrameResult:
        t = time.perf_counter() if t is None else t
        t0 = time.perf_counter()
        pose = self.tracker(img_bgr)
        t1 = time.perf_counter()
        return self._after_2d(pose.keypoints, pose.box, pose.visible, t, {"pose2d_ms": 1000 * (t1 - t0)})

    def process_keypoints(self, kp_coco: np.ndarray, t: float, box: np.ndarray | None = None) -> FrameResult:
        """Entry point when 2D keypoints come from elsewhere (tests, a browser-side pose model)."""
        return self._after_2d(kp_coco, box, bool(kp_coco[:, 2].mean() > 0.2), t, {})

    def _after_2d(self, kp: np.ndarray, box, visible: bool, t: float, timings: dict) -> FrameResult:
        f = self.frame
        self.frame += 1
        raw = kp
        kp = kp.copy()
        if visible:
            kp[:, :2] = self.filter(kp[:, :2], t, conf=kp[:, 2]).astype(np.float32)
        t1 = time.perf_counter()
        # The filtered 2D is what gets drawn; the lifter gets the raw keypoints unless asked otherwise - the
        # 2D filter's lag costs joint-angle accuracy, and the 3D filter removes the jitter (BENCHMARK.md).
        final = self.stream.push(f, coco_to_h36m(kp if self.filter_2d_for_lift else raw))
        t2 = time.perf_counter()
        upd = None
        merged = CoachUpdate()
        for fr, pose3d in final:
            if self.filter3d is not None:
                # causal lifting jitters; smooth the skeleton on the camera's clock
                pose3d = self.filter3d(pose3d, t - (self.frame - 1 - fr) / max(self.coach.fps, 1.0)).astype(np.float32)
            upd = self.coach.ingest(fr, t, pose3d)
            merged.reps += upd.reps
            merged.partials += upd.partials
            merged.cues += upd.cues
        if upd is not None:
            upd.reps, upd.partials, upd.cues = merged.reps, merged.partials, merged.cues
            self._last = upd
        else:
            upd = CoachUpdate(**{**self._last.__dict__, "reps": [], "partials": [], "cues": []})
        timings |= {"lift_ms": 1000 * (t2 - t1), "coach_ms": 1000 * (time.perf_counter() - t2)}
        return FrameResult(f, t, kp, box, visible, upd, timings)
