"""YAML configuration: ``configs/default.yaml`` plus ``key.sub=value`` overrides from the command line."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .paths import CONFIG_DIR


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return set_dotted(cfg, overrides or [])


def set_dotted(cfg: dict, assignments: list[str]) -> dict:
    """Apply ``live.window=27``-style overrides."""
    for a in assignments:
        key, sep, val = a.partition("=")
        if not sep:
            raise ValueError(f"override {a!r} must look like key=value")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def live_settings(cfg: dict, device: str) -> dict:
    """The ``live`` block, with ``live_cpu`` applied on a CPU-only machine."""
    out = dict(cfg.get("live", {}))
    if device != "cuda":
        out.update(cfg.get("live_cpu", {}) or {})
    return out
