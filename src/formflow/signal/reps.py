"""Online repetition segmentation from a 1-D movement signal (e.g. mean knee flexion for a squat).

The signal is normalised to a 0..1 "excursion" ``p`` (0 = start/lockout position, 1 = the far end of
the movement), then a hysteresis state machine walks through the phases of every rep:

    top (lockout) --p > enter--> down --p >= bottom--> bottom --p < peak - reversal--> up --p < exit--> top
                                     \\____________ shallow turn-around ____________/

A rep is counted on the return to ``top`` if its peak excursion reached ``depth_min``; shallower
turn-arounds are reported as partial attempts (not counted). Nothing is counted until the athlete has
been seen at the start position once, so a video that begins mid-rep does not produce a phantom rep. Normalisation is adaptive: the 5th/95th
percentiles of the last few seconds give the athlete's own range of motion, with the exercise's
nominal range (``prior``) as a fallback until enough motion has been seen. ``min_rom`` stops
standing still or fidgeting from being normalised up into fake reps.

The same code runs live (one sample at a time) and offline (``segment``), so the numbers in the
report match what the live coach showed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

PHASES = ("top", "down", "bottom", "up")


@dataclass
class RepParams:
    prior: tuple[float, float] | None = None  # nominal (start, far-end) values of the signal
    min_rom: float = 25.0  # smallest range (signal units) that counts as movement
    window_s: float = 8.0  # adaptive range: percentiles over this many seconds
    enter: float = 0.30  # leave the top when p rises above this
    exit: float = 0.25  # back at the top (rep complete) when p falls below this
    bottom: float = 0.80  # "bottom" phase when p is above this
    reversal: float = 0.15  # turn-around: p dropped this far below the rep's peak
    depth_min: float = 0.55  # a turn-around shallower than this is a partial rep, not counted
    min_rep_s: float = 0.35
    max_rep_s: float = 15.0
    adaptive: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> RepParams:
        d = dict(d or {})
        if d.get("prior") is not None:
            d["prior"] = tuple(d["prior"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Rep:
    index: int  # 1-based count (0 for partial attempts)
    start_frame: int
    bottom_frame: int
    end_frame: int
    t_start: float
    t_bottom: float
    t_end: float
    peak_value: float  # raw signal at the far end
    start_value: float  # raw signal when the rep began
    excursion: float  # normalised peak depth (0..1+)
    counted: bool = True
    faults: list = field(default_factory=list)
    score: float | None = None

    @property
    def out_s(self) -> float:
        return self.t_bottom - self.t_start

    @property
    def back_s(self) -> float:
        return self.t_end - self.t_bottom

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(out_s=round(self.out_s, 3), back_s=round(self.back_s, 3),
                 duration_s=round(self.duration_s, 3))
        return d


class RepCounter:
    def __init__(self, params: RepParams | None = None) -> None:
        self.p = params or RepParams()
        self.reset()

    def reset(self) -> None:
        self.state = "top"
        self.count = 0
        self.reps: list[Rep] = []
        self.partials: list[Rep] = []
        self._hist: deque[tuple[float, float]] = deque()
        self._start: tuple[int, float, float] | None = None  # (frame, t, value) where the rep began
        self._last_top: tuple[int, float, float] | None = None  # most extended point while at the top
        self._peak: tuple[int, float, float] | None = None  # far end of the current rep (raw value)
        self._trough: tuple[int, float, float] | None = None  # lowest point since the turn-around
        self._settle: tuple[int, float, float] | None = None  # back at the top, waiting for the lockout
        self._armed = False  # the start position has been seen (a rep already under way when we start is not judged)
        self._first: float | None = None  # first value ever seen
        self.level = (np.nan, np.nan)
        self.excursion = 0.0

    # ---- normalisation -------------------------------------------------------
    def _levels(self) -> tuple[float, float] | None:
        p = self.p
        lo_hi = None
        if p.adaptive and len(self._hist) >= 5:
            vals = np.fromiter((v for _, v in self._hist), float)
            lo, hi = np.percentile(vals, [5, 95])
            if hi - lo >= p.min_rom:
                lo_hi = (lo, hi)
                if p.prior is not None and p.prior[0] > p.prior[1]:
                    lo_hi = (hi, lo)  # the movement goes toward smaller values
        if lo_hi is None and p.prior is not None:
            lo_hi = p.prior
        return lo_hi

    def _norm(self, v: float, lv: tuple[float, float]) -> float:
        return float((v - lv[0]) / (lv[1] - lv[0])) if lv[1] != lv[0] else 0.0

    # ---- step ------------------------------------------------------------------
    def update(self, frame: int, t: float, value: float) -> Rep | None:
        """Feed one sample. Returns the Rep when one is completed on this sample (counted or partial).

        Peaks and troughs are kept as raw signal values and re-normalised every sample, because the
        adaptive range is still widening during the first rep.
        """
        prm = self.p
        if not np.isfinite(value):
            return None
        self._hist.append((t, value))
        if self._first is None:
            self._first = value
        while self._hist and t - self._hist[0][0] > prm.window_s:
            self._hist.popleft()
        lv = self._levels()
        if lv is None:
            self.excursion = 0.0
            return None
        self.level = lv
        nx = lambda v: self._norm(v, lv)
        x = nx(value)
        self.excursion = x
        done: Rep | None = None

        if self.state == "top":
            if self._settle is not None:
                # Back near the top: the rep ends at the lockout - where the signal stops returning.
                if x < nx(self._settle[2]) - 0.01:  # strictly lower: the first frame of the lockout wins
                    self._settle = (frame, t, value)
                if x > nx(self._settle[2]) + 0.05 or t - self._settle[1] >= 0.25 or x > prm.enter:
                    done = self._finish(*self._settle)
            if self._last_top is None or x <= nx(self._last_top[2]):
                self._last_top = (frame, t, value)
            # Armed once the athlete is seen at the start position - or started there (the range may only
            # become known part-way through the first rep, e.g. in auto mode without a nominal range).
            if x < prm.exit or (self._first is not None and nx(self._first) < prm.exit):
                self._armed = True
            if x > prm.enter and self._armed:
                self.state = "down"
                self._start = self._last_top if self._last_top else (frame, t, value)
                self._peak = (frame, t, value)
            return done

        assert self._start is not None and self._peak is not None
        if x > nx(self._peak[2]):
            self._peak = (frame, t, value)
        px = nx(self._peak[2])
        if self.state == "down" and x >= prm.bottom:
            self.state = "bottom"
        if self.state in ("down", "bottom") and x < px - prm.reversal:
            self.state = "up"
            self._trough = (frame, t, value)
        elif self.state == "up":
            assert self._trough is not None
            if x > nx(self._trough[2]) + prm.reversal:
                # Bounced back down before reaching the top: the same rep continues.
                self.state = "down" if x < prm.bottom else "bottom"
            elif x < nx(self._trough[2]):
                self._trough = (frame, t, value)
        if self.state == "up" and x < prm.exit:
            self.state = "top"
            self._settle = (frame, t, value)
            self._last_top = (frame, t, value)
        elif t - self._start[1] > prm.max_rep_s:
            self.state = "top"  # stalled or lost track: start over
            self._last_top = (frame, t, value)
        return done

    def _finish(self, frame: int, t: float, value: float) -> Rep | None:
        assert self._start is not None and self._peak is not None
        prm = self.p
        sf, st, sv = self._start
        pf, pt, pv = self._peak
        self._settle = None
        if t - st < prm.min_rep_s:
            return None
        px = self._norm(pv, self.level)
        counted = px >= prm.depth_min
        rep = Rep(index=self.count + 1 if counted else 0, start_frame=sf, bottom_frame=pf, end_frame=frame,
                  t_start=st, t_bottom=pt, t_end=t, peak_value=float(pv), start_value=float(sv),
                  excursion=float(px), counted=counted)
        if counted:
            self.count += 1
            self.reps.append(rep)
        else:
            self.partials.append(rep)
        return rep

    def flush(self) -> Rep | None:
        """End of input: finish a rep that is waiting for its lockout."""
        if self._settle is not None and self.state == "top":
            return self._finish(*self._settle)
        return None

    @property
    def phase(self) -> str:
        return self.state


def segment(signal: np.ndarray, fps: float, params: RepParams | None = None) -> tuple[list[Rep], list[str]]:
    """Offline: run the online counter over a whole signal. Returns (counted reps, per-frame phase)."""
    rc = RepCounter(params)
    phases = []
    for i, v in enumerate(np.asarray(signal, float)):
        rc.update(i, i / fps, v)
        phases.append(rc.phase)
    rc.flush()
    return rc.reps, phases


def periodicity(signal: np.ndarray, fps: float, min_period_s: float = 0.4, max_period_s: float = 8.0) -> float:
    """How repetitive a signal is: the autocorrelation peak within a plausible rep period, times its range.

    Used by the automatic mode to decide which joint angle carries the reps. The range factor makes a
    large, clean oscillation beat a tiny one that happens to be regular.
    """
    x = np.asarray(signal, float)
    x = x[np.isfinite(x)]
    n = len(x)
    lo, hi = int(min_period_s * fps), min(int(max_period_s * fps), n // 2)
    if n < 8 or hi <= lo + 1:
        return 0.0
    rng = float(np.percentile(x, 95) - np.percentile(x, 5))
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    return float(max(ac[lo:hi].max(), 0.0) * rng)
