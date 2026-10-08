"""Shared test helpers (the tests folder is on sys.path under pytest's default import mode)."""

import torch

from formflow.lift.dstformer import DSTformer


def tiny_model(seed: int = 0, maxlen: int = 81) -> DSTformer:
    """A small randomly initialised DSTformer: exercises every code path without the 170 MB checkpoint."""
    torch.manual_seed(seed)
    m = DSTformer(dim_feat=32, dim_rep=32, depth=2, num_heads=4, mlp_ratio=2, maxlen=maxlen)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()
