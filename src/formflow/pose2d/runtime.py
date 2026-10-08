"""ONNX Runtime sessions with the best available accelerator.

On an NVIDIA machine the CUDA execution provider needs the CUDA / cuDNN libraries. The PyTorch CUDA
wheel FormFlow already installs ships exactly those, so importing torch first (which loads them into
the process) is all onnxruntime-gpu needs: no separate CUDA toolkit install.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def available_providers() -> tuple[str, ...]:
    import onnxruntime as ort

    return tuple(ort.get_available_providers())


def _prepare_cuda() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:  # torch missing or broken: ORT may still find a system CUDA
        pass
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")  # reuse the libraries torch loaded (no-op if already present)
    except Exception:
        pass
    return True


def make_session(path: str, device: str = "auto"):
    """``device``: auto | cuda | cpu. Falls back to the CPU if the GPU provider fails to initialise."""
    import onnxruntime as ort

    ort.set_default_logger_severity(3)  # keep provider probing quiet; failures fall back to the CPU below
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    threads = int(os.environ.get("FORMFLOW_CPU_THREADS", "0"))
    if threads:
        opts.intra_op_num_threads = threads
    provs: list = ["CPUExecutionProvider"]
    avail = available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail and _prepare_cuda():
        provs.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
    elif device in ("auto", "dml") and "DmlExecutionProvider" in avail:
        provs.insert(0, "DmlExecutionProvider")
    try:
        sess = ort.InferenceSession(path, sess_options=opts, providers=provs)
    except Exception:
        sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return sess


def session_device(sess) -> str:
    p = sess.get_providers()[0]
    return {"CUDAExecutionProvider": "cuda", "DmlExecutionProvider": "dml"}.get(p, "cpu")
