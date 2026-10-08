"""Exercise definitions, loaded from ``configs/exercises/*.yaml``.

Each file names the signal that carries the reps, the nominal range of that signal, and the form
rules. Adding an exercise means adding a YAML file: no code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..paths import CONFIG_DIR
from ..signal.reps import RepParams

RULE_TYPES = ("limit", "tempo", "ratio")
AGGS = ("min", "max", "mean", "median", "p10", "p90", "start", "end", "bottom", "range", "rise")
WINDOWS = ("rep", "down", "bottom", "up", "top")


@dataclass
class Rule:
    id: str
    fault: str
    cue: str
    type: str = "limit"
    metric: str | None = None
    agg: str = "max"
    phase: list[str] | str = "rep"
    min: float | None = None
    max: float | None = None
    scale: float = 1.0
    weight: float = 10.0
    joints: list[str] = field(default_factory=list)
    sides: list[str] = field(default_factory=list)
    live: bool = False
    a: str | None = None
    b: str | None = None
    frac: float = 0.4
    view: str = "any"  # "front": only judged when the camera is not side-on (symmetry, frontal-plane rules)

    def __post_init__(self) -> None:
        if self.type not in RULE_TYPES:
            raise ValueError(f"rule {self.id}: unknown type {self.type!r} (one of {RULE_TYPES})")
        if self.type == "limit":
            if not self.metric:
                raise ValueError(f"rule {self.id}: a limit rule needs a metric")
            if self.agg not in AGGS:
                raise ValueError(f"rule {self.id}: unknown agg {self.agg!r} (one of {AGGS})")
            for w in self.windows:
                if w not in WINDOWS:
                    raise ValueError(f"rule {self.id}: unknown phase {w!r} (one of {WINDOWS})")
        if self.type == "ratio" and not (self.a and self.b):
            raise ValueError(f"rule {self.id}: a ratio rule needs metrics a and b")
        if self.type == "tempo" and self.phase not in ("out", "back", "total"):
            raise ValueError(f"rule {self.id}: tempo phase must be out, back or total")
        if self.min is None and self.max is None:
            raise ValueError(f"rule {self.id}: needs min and/or max")
        if self.view not in ("any", "front"):
            raise ValueError(f"rule {self.id}: view must be any or front")
        if self.scale <= 0:
            raise ValueError(f"rule {self.id}: scale must be > 0")

    @property
    def windows(self) -> list[str]:
        return [self.phase] if isinstance(self.phase, str) else list(self.phase)


@dataclass
class Exercise:
    name: str
    display: str
    signal: str
    reps: RepParams
    rules: list[Rule]
    posture: str = "standing"
    camera: str = ""
    aliases: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)

    @property
    def auto(self) -> bool:
        return self.signal == "auto"

    def to_dict(self) -> dict:
        return {"name": self.name, "display": self.display, "signal": self.signal, "posture": self.posture,
                "camera": self.camera, "rules": [{"id": r.id, "fault": r.fault, "cue": r.cue} for r in self.rules],
                "metrics": list(dict.fromkeys(m for r in self.rules for m in (r.metric, r.a, r.b) if m))}


def parse_exercise(d: dict) -> Exercise:
    rules = [Rule(**r) for r in d.get("rules") or []]
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError(f"exercise {d.get('name')}: duplicate rule ids")
    return Exercise(name=d["name"], display=d.get("display", d["name"]), signal=d["signal"],
                    reps=RepParams.from_dict(d.get("reps")), rules=rules, posture=d.get("posture", "standing"),
                    camera=d.get("camera", ""), aliases=list(d.get("aliases") or []),
                    candidates=list(d.get("candidates") or []))


@lru_cache(maxsize=4)
def _load_all(folder: str) -> dict[str, Exercise]:
    out: dict[str, Exercise] = {}
    for f in sorted(Path(folder).glob("*.yaml")):
        ex = parse_exercise(yaml.safe_load(f.read_text(encoding="utf-8")))
        out[ex.name] = ex
    return out


def exercises(folder: Path | None = None) -> dict[str, Exercise]:
    return _load_all(str(folder or CONFIG_DIR / "exercises"))


def get_exercise(name: str, folder: Path | None = None) -> Exercise:
    """Look an exercise up by name or alias (case / dash / space insensitive)."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    all_ = exercises(folder)
    if key in all_:
        return all_[key]
    for ex in all_.values():
        if key in ex.aliases:
            return ex
    raise KeyError(f"unknown exercise {name!r}; known: {', '.join(sorted(all_))}")
