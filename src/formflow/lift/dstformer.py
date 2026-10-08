"""DSTformer, the dual-stream spatio-temporal transformer of MotionBERT.

Ported from https://github.com/Walter0807/MotionBERT (``lib/model/DSTformer.py``, Apache-2.0,
Zhu et al., "MotionBERT: A Unified Perspective on Learning Human Motion Representations", ICCV 2023).
Parameter names are unchanged so the official checkpoints load with ``strict=True``. Changes: only the
inference path is kept, and attention uses ``torch.nn.functional.scaled_dot_product_attention``
(fused kernels; same maths as the original softmax(q k^T * scale) v).

Input  (B, F, 17, 3): H36M-ordered 2D keypoints (x, y normalised to about [-1, 1]) + confidence.
Output (B, F, 17, 3): 3D joints in the same normalised units (camera frame, y down).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention over joints ("spatial") or over frames for each joint ("temporal")."""

    def __init__(self, dim: int, num_heads: int, mode: str, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, seqlen: int) -> torch.Tensor:
        BF, N, C = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(BF, N, 3, H, C // H).permute(2, 0, 3, 1, 4)  # (3, BF, H, N, c)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.mode == "spatial":
            o = F.scaled_dot_product_attention(q, k, v)  # (BF, H, N, c)
            o = o.transpose(1, 2).reshape(BF, N, C)
        else:  # temporal: attend across the F frames of each joint
            def t(z: torch.Tensor) -> torch.Tensor:
                return z.reshape(-1, seqlen, H, N, C // H).permute(0, 2, 3, 1, 4)  # (B, H, N, F, c)

            o = F.scaled_dot_product_attention(t(q), t(k), t(v))
            o = o.permute(0, 3, 2, 1, 4).reshape(BF, N, C)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, st_mode: str) -> None:
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s = nn.LayerNorm(dim)
        self.norm1_t = nn.LayerNorm(dim)
        self.attn_s = Attention(dim, num_heads, "spatial")
        self.attn_t = Attention(dim, num_heads, "temporal")
        self.norm2_s = nn.LayerNorm(dim)
        self.norm2_t = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_s = MLP(dim, hidden, dim)
        self.mlp_t = MLP(dim, hidden, dim)

    def _s(self, x: torch.Tensor, seqlen: int) -> torch.Tensor:
        x = x + self.attn_s(self.norm1_s(x), seqlen)
        return x + self.mlp_s(self.norm2_s(x))

    def _t(self, x: torch.Tensor, seqlen: int) -> torch.Tensor:
        x = x + self.attn_t(self.norm1_t(x), seqlen)
        return x + self.mlp_t(self.norm2_t(x))

    def forward(self, x: torch.Tensor, seqlen: int) -> torch.Tensor:
        if self.st_mode == "stage_st":
            return self._t(self._s(x, seqlen), seqlen)
        return self._s(self._t(x, seqlen), seqlen)


class DSTformer(nn.Module):
    def __init__(self, dim_in: int = 3, dim_out: int = 3, dim_feat: int = 512, dim_rep: int = 512, depth: int = 5,
                 num_heads: int = 8, mlp_ratio: float = 2, num_joints: int = 17, maxlen: int = 243) -> None:
        super().__init__()
        self.maxlen = maxlen
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.blocks_st = nn.ModuleList([Block(dim_feat, num_heads, mlp_ratio, "stage_st") for _ in range(depth)])
        self.blocks_ts = nn.ModuleList([Block(dim_feat, num_heads, mlp_ratio, "stage_ts") for _ in range(depth)])
        self.norm = nn.LayerNorm(dim_feat)
        self.pre_logits = nn.Sequential(OrderedDict([("fc", nn.Linear(dim_feat, dim_rep)), ("act", nn.Tanh())]))
        self.head = nn.Linear(dim_rep, dim_out)
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Fr, J, C = x.shape
        if Fr > self.maxlen:
            raise ValueError(f"at most {self.maxlen} frames per clip (got {Fr})")
        x = self.joints_embed(x.reshape(-1, J, C)) + self.pos_embed
        x = (x.reshape(B, Fr, J, -1) + self.temp_embed[:, :Fr]).reshape(B * Fr, J, -1)
        for blk_st, blk_ts, att in zip(self.blocks_st, self.blocks_ts, self.ts_attn):
            x_st, x_ts = blk_st(x, Fr), blk_ts(x, Fr)
            alpha = att(torch.cat([x_st, x_ts], dim=-1)).softmax(dim=-1)
            x = x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]
        x = self.norm(x).reshape(B, Fr, J, -1)
        return self.head(self.pre_logits(x))


# Architectures of the released 3D-pose checkpoints (configs/pose3d/*.yaml in the MotionBERT repo).
VARIANTS = {
    "full": {"dim_feat": 512, "mlp_ratio": 2, "depth": 5, "dim_rep": 512, "num_heads": 8, "rootrel": True},
    "lite": {"dim_feat": 256, "mlp_ratio": 4, "depth": 5, "dim_rep": 512, "num_heads": 8, "rootrel": False},
}


def build(variant: str) -> DSTformer:
    cfg = {k: v for k, v in VARIANTS[variant].items() if k != "rootrel"}
    return DSTformer(**cfg)


def load_checkpoint(model: DSTformer, path: str) -> DSTformer:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_pos", ck)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model
