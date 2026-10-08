"""MotionBERT port, normalisation, streaming == batched causal lifting, SimCC decoding, tracking."""

import numpy as np
import pytest
import torch

from formflow.bench.lift_eval import causal_lift, mpjpe, pa_mpjpe
from formflow.lift.lifter import Lifter, StreamingLifter, fill_missing, normalise
from formflow.pose2d.models import decode_simcc
from formflow.pose2d.tracker import PoseTracker, box_from_keypoints
from formflow.skeleton import H36M_FLIP
from formflow.synth import SquatSpec, project, squat_sequence
from helpers import tiny_model


def reference_attention(attn, x, seqlen):
    """The original MotionBERT maths (explicit softmax(q k^T * scale) v), for checking the SDPA port."""
    B, N, C = x.shape
    H = attn.num_heads
    qkv = attn.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scale = (C // H) ** -0.5
    if attn.mode == "spatial":
        a = (q @ k.transpose(-2, -1) * scale).softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(B, N, C)
    else:
        t = lambda z: z.reshape(-1, seqlen, H, N, C // H).permute(0, 2, 3, 1, 4)
        qt, kt, vt = t(q), t(k), t(v)
        a = (qt @ kt.transpose(-2, -1) * scale).softmax(-1)
        o = (a @ vt).permute(0, 3, 2, 1, 4).reshape(B, N, C)
    return attn.proj(o)


def test_dstformer_attention_matches_the_original_formulation():
    m = tiny_model()
    x = torch.randn(2 * 9, 17, 32)
    for blk in m.blocks_st:
        for att in (blk.attn_s, blk.attn_t):
            torch.testing.assert_close(att(x, 9), reference_attention(att, x, 9), atol=1e-5, rtol=1e-4)
    y = m(torch.randn(2, 9, 17, 3))
    assert y.shape == (2, 9, 17, 3)
    with pytest.raises(ValueError):
        m(torch.randn(1, 82, 17, 3))


def test_lifter_flip_is_consistent_and_root_relative():
    lf = Lifter(None, model=tiny_model(), device="cpu", flip=True)
    x = np.random.default_rng(0).normal(0, 0.3, (1, 27, 17, 3)).astype(np.float32)
    x[..., 2] = 1.0
    y = lf.infer(x)
    np.testing.assert_allclose(y[:, :, 0], 0, atol=1e-6)
    xf = x.copy()
    xf[..., 0] *= -1
    xf = xf[:, :, H36M_FLIP]
    yf = lf.infer(xf)
    yf[..., 0] *= -1
    np.testing.assert_allclose(yf[:, :, H36M_FLIP], y, atol=1e-4)  # flip-averaging makes it exactly equivariant


def test_normalise_round_trip_and_scale_invariance():
    seq, _ = squat_sequence(SquatSpec(reps=1))
    kp = project(seq)[:40]
    for mode in ("crop", "image"):
        x, nm = normalise(kp, (1280, 720), mode)
        back = nm.denorm(np.concatenate([x[..., :2], np.zeros_like(x[..., :1])], -1))
        np.testing.assert_allclose(back[..., :2], kp[..., :2], atol=1e-3)
    big = kp.copy()
    big[..., :2] = (big[..., :2] - 400) * 1.7 + 400
    a, _ = normalise(kp, (1280, 720), "crop")
    b, _ = normalise(big, (1280, 720), "crop")
    np.testing.assert_allclose(a, b, atol=1e-4)  # crop mode: zoom does not change the network input


def test_streaming_lifter_equals_batched_causal_lifting():
    lf = Lifter(None, model=tiny_model(), device="cpu", flip=False)
    seq, _ = squat_sequence(SquatSpec(reps=1))
    kp = project(seq)[:50]
    for stride in (1, 3):
        st = StreamingLifter(lf, (1280, 720), window=27, stride=stride)
        out = np.zeros((len(kp), 17, 3), np.float32)
        for i, k in enumerate(kp):
            for f, p in st.push(i, k, force=(i == len(kp) - 1)):
                out[f] = p
        ref = causal_lift(lf, kp, (1280, 720), 27, stride)
        np.testing.assert_allclose(out, ref, atol=1e-4)


def test_offline_lifting_covers_every_frame_and_fill_missing():
    lf = Lifter(None, model=tiny_model(), device="cpu", flip=False)
    seq, _ = squat_sequence(SquatSpec(reps=2))
    kp = project(seq)
    kp[10:15, :, 2] = 0  # nobody detected for 5 frames
    filled = fill_missing(kp)
    np.testing.assert_allclose(filled[12, :, :2], kp[9, :, :2])
    y = lf.lift_sequence(kp, (1280, 720), window=81, overlap=0.5)
    assert y.shape == kp.shape and np.isfinite(y).all()


def test_pose_errors():
    rng = np.random.default_rng(0)
    g = rng.normal(0, 100, (4, 17, 3))
    assert mpjpe(g, g).max() == 0
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    p = 2.0 * g @ R.T + 5
    assert pa_mpjpe(p, g).max() < 1e-6  # similarity transforms are removed
    assert mpjpe(p, g).min() > 10


def test_simcc_decoding():
    sx = np.zeros((1, 17, 384), np.float32)
    sy = np.zeros((1, 17, 512), np.float32)
    sx[0, :, 100] = 0.9
    sy[0, :, 300] = 0.7
    locs, vals = decode_simcc(sx, sy)
    np.testing.assert_allclose(locs[0, 0], [50, 150])
    np.testing.assert_allclose(vals, 0.7)


class FakeDet:
    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        return np.array([[100, 50, 300, 450, 0.9]], np.float32)


class FakePose:
    def __init__(self, conf=0.9):
        self.conf = conf

    def __call__(self, img, boxes):
        b = boxes[0]
        xs = np.linspace(b[0] + 20, b[2] - 20, 17)
        ys = np.linspace(b[1] + 20, b[3] - 20, 17)
        return np.stack([xs, ys, np.full(17, self.conf)], -1)[None].astype(np.float32)


def test_tracker_detects_rarely_and_recovers_when_lost():
    det, pose = FakeDet(), FakePose()
    tr = PoseTracker(det, pose, det_every=5)
    img = np.zeros((480, 640, 3), np.uint8)
    for _ in range(12):
        p = tr(img)
    assert det.calls == 3 and p.visible  # frames 0, 5, 10
    pose.conf = 0.1  # the athlete is lost: detect immediately
    tr(img)
    assert det.calls == 4
    assert box_from_keypoints(np.zeros((17, 3), np.float32), (640, 480)) is None
