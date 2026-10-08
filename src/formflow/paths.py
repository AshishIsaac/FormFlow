"""Where things live. Everything downloaded or produced goes under FORMFLOW_HOME (default: ./workspace)."""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
CONFIG_DIR = Path(os.environ.get("FORMFLOW_CONFIGS", REPO_ROOT / "configs"))
REFERENCE_DIR = REPO_ROOT / "benchmarks" / "results"
SAMPLES_DIR = REPO_ROOT / "samples"


def home() -> Path:
    return Path(os.environ.get("FORMFLOW_HOME", REPO_ROOT / "workspace")).expanduser().resolve()


def models_dir() -> Path:
    return home() / "models"


def data_dir() -> Path:
    return home() / "data"


def sessions_dir() -> Path:
    """Analysed videos and recorded live sessions (annotated video, report, raw tracks)."""
    return home() / "sessions"


def bench_dir() -> Path:
    return home() / "bench"
