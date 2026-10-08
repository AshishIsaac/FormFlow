"""Drawing: the annotated frame used by the desktop app and by annotated videos.

Layout (all OpenCV, no GUI toolkit):

    +--------------------------------------+----------------+
    | camera frame + 2D skeleton           |  3D skeleton   |
    | (faulty joints red)                  |  (side view)   |
    |  REPS 7   phase pill   last score    |----------------|
    |                                      |  joint angles  |
    | [ cue banner ]                       |                |
    +--------------------------------------+----------------+
    | signal strip: the rep signal over the last seconds, reps marked              |
    +------------------------------------------------------------------------------+
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..skeleton import COCO_BONES, COCO_SIDE, H36M_BONES, H36M_SIDE

FONT = cv2.FONT_HERSHEY_DUPLEX
BG = (24, 22, 20)
PANEL = (38, 34, 31)
FG = (235, 235, 235)
MUTED = (150, 145, 140)
LEFT = (235, 160, 60)  # BGR: blue-ish for the athlete's left
RIGHT = (60, 150, 245)  # orange for the right
CENTRE = (200, 200, 200)
BAD = (70, 70, 235)
GOOD = (110, 200, 90)
WARN = (40, 190, 240)
PHASE_COL = {"top": (150, 150, 150), "down": (235, 160, 60), "bottom": (60, 150, 245), "up": (110, 200, 90)}

# H36M joint names used by rules -> COCO keypoints to highlight on the 2D overlay.
H36M_TO_COCO = {"l_knee": [13], "r_knee": [14], "l_hip": [11], "r_hip": [12], "l_ankle": [15], "r_ankle": [16],
                "l_elbow": [7], "r_elbow": [8], "l_wrist": [9], "r_wrist": [10], "l_shoulder": [5],
                "r_shoulder": [6], "pelvis": [11, 12], "spine": [5, 6, 11, 12], "thorax": [5, 6], "neck": [0],
                "head": [0, 1, 2]}
H36M_INDEX = {n: i for i, n in enumerate(("pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "spine",
                                          "thorax", "neck", "head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder",
                                          "r_elbow", "r_wrist"))}


def _side_col(s: int) -> tuple[int, int, int]:
    return LEFT if s > 0 else RIGHT if s < 0 else CENTRE


def text(img, s: str, org, scale=0.6, col=FG, thick=1, anchor="tl") -> tuple[int, int]:
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if anchor[1] == "r":
        x -= w
    elif anchor[1] == "c":
        x -= w // 2
    if anchor[0] == "t":
        y += h
    elif anchor[0] == "c":
        y += h // 2
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, col, thick, cv2.LINE_AA)
    return w, h + base


def draw_skeleton_2d(img: np.ndarray, kp: np.ndarray, bad_joints: set[int] | None = None, thr: float = 0.3,
                     scale: float = 1.0) -> None:
    bad_joints = bad_joints or set()
    t = max(2, int(3 * scale))
    for a, b in COCO_BONES:
        if kp[a, 2] < thr or kp[b, 2] < thr:
            continue
        col = BAD if (a in bad_joints and b in bad_joints) else _side_col(int(COCO_SIDE[b]))
        cv2.line(img, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), col, t, cv2.LINE_AA)
    for j in range(17):
        if kp[j, 2] < thr:
            continue
        p = tuple(int(v) for v in kp[j, :2])
        if j in bad_joints:
            cv2.circle(img, p, int(11 * scale), BAD, max(2, int(2 * scale)), cv2.LINE_AA)
        cv2.circle(img, p, max(3, int(4 * scale)), FG if j not in bad_joints else BAD, -1, cv2.LINE_AA)


def _rot(yaw: float, pitch: float) -> np.ndarray:
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rx @ ry


def level_rotation(up: np.ndarray | None) -> np.ndarray:
    """Rotation taking the estimated gravity-up vector to the camera's -y axis (so the floor is level)."""
    if up is None:
        return np.eye(3)
    a = up / max(np.linalg.norm(up), 1e-9)
    b = np.array([0.0, -1.0, 0.0])
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def draw_skeleton_3d(panel: np.ndarray, pose: np.ndarray | None, yaw: float = -60.0, pitch: float = 12.0,
                     bad: set[int] | None = None, label: str = "3D (side view)", up: np.ndarray | None = None) -> None:
    """Orthographic view of the lifted skeleton, levelled to gravity, with a floor grid."""
    h, w = panel.shape[:2]
    panel[:] = PANEL
    text(panel, label, (10, 8), 0.45, MUTED)
    if pose is None:
        text(panel, "waiting for the athlete...", (w // 2, h // 2), 0.5, MUTED, anchor="cc")
        return
    bad = bad or set()
    pose = (pose - pose[:1]) @ level_rotation(up).T
    R = _rot(yaw, pitch)
    p = pose @ R.T
    extent = max(float(np.ptp(p[:, 1])), float(np.ptp(p[:, 0])), 1e-6)
    s = 0.72 * min(w, h) / extent
    c = np.array([w / 2, h * 0.52])
    xy = p[:, :2] * s + c - np.array([0, (p[:, 1].max() + p[:, 1].min()) / 2 * s])
    # floor grid at the feet
    foot_y = float(max(pose[3, 1], pose[6, 1]) - pose[0, 1])
    for gx in np.linspace(-1, 1, 7):
        for a_, b_ in (((gx, -1), (gx, 1)), ((-1, gx), (1, gx))):
            q = np.array([[a_[0], foot_y / 1, a_[1]], [b_[0], foot_y / 1, b_[1]]], float)
            q[:, [0, 2]] *= extent * 0.6
            q = q @ R.T
            qq = q[:, :2] * s + c - np.array([0, (p[:, 1].max() + p[:, 1].min()) / 2 * s])
            cv2.line(panel, tuple(qq[0].astype(int)), tuple(qq[1].astype(int)), (60, 56, 52), 1, cv2.LINE_AA)
    order = sorted(range(len(H36M_BONES)), key=lambda k: -(p[H36M_BONES[k][0], 2] + p[H36M_BONES[k][1], 2]))
    for k in order:
        a, b = H36M_BONES[k]
        col = BAD if (a in bad or b in bad) else _side_col(int(H36M_SIDE[b]))
        cv2.line(panel, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), col, 4, cv2.LINE_AA)
    for j in range(17):
        cv2.circle(panel, tuple(xy[j].astype(int)), 4, BAD if j in bad else FG, -1, cv2.LINE_AA)


class SignalStrip:
    """Rolling plot of the rep signal with completed reps marked."""

    def __init__(self, seconds: float = 12.0) -> None:
        self.seconds = seconds
        self.buf: deque[tuple[float, float, str]] = deque()
        self.marks: deque[tuple[float, int, float | None]] = deque()

    def add(self, t: float, v: float | None, phase: str) -> None:
        if v is None or not np.isfinite(v):
            return
        self.buf.append((t, v, phase))
        while self.buf and t - self.buf[0][0] > self.seconds:
            self.buf.popleft()
        while self.marks and t - self.marks[0][0] > self.seconds:
            self.marks.popleft()

    def mark(self, t: float, index: int, score: float | None) -> None:
        self.marks.append((t, index, score))

    def draw(self, strip: np.ndarray, label: str) -> None:
        h, w = strip.shape[:2]
        strip[:] = PANEL
        text(strip, label, (10, 6), 0.42, MUTED)
        if len(self.buf) < 2:
            return
        t_end = self.buf[-1][0]
        ts = np.array([b[0] for b in self.buf])
        vs = np.array([b[1] for b in self.buf])
        lo, hi = float(vs.min()), float(vs.max())
        if hi - lo < 10:
            mid = (hi + lo) / 2
            lo, hi = mid - 5, mid + 5
        x = (w - 10) - (t_end - ts) / self.seconds * (w - 20)
        y = h - 8 - (vs - lo) / (hi - lo) * (h - 30)
        for i in range(1, len(x)):
            col = PHASE_COL.get(self.buf[i][2], FG)
            cv2.line(strip, (int(x[i - 1]), int(y[i - 1])), (int(x[i]), int(y[i])), col, 2, cv2.LINE_AA)
        for t, idx, score in self.marks:
            mx = int((w - 10) - (t_end - t) / self.seconds * (w - 20))
            col = GOOD if (score or 0) >= 80 else WARN if (score or 0) >= 60 else BAD
            cv2.line(strip, (mx, 22), (mx, h - 4), col, 1, cv2.LINE_AA)
            text(strip, f"{idx}", (mx + 3, 22), 0.42, col)


def _bad_sets(live: list[dict]) -> tuple[set[int], set[int]]:
    coco, h36 = set(), set()
    for v in live:
        for j in v.get("joints", []):
            coco.update(H36M_TO_COCO.get(j, []))
            if j in H36M_INDEX:
                h36.add(H36M_INDEX[j])
    return coco, h36


class Renderer:
    """Composes the full annotated view for one frame."""

    def __init__(self, exercise_display: str, angle_keys: list[tuple[str, str]], side_w: int = 340,
                 strip_h: int = 120, target_h: int = 720) -> None:
        self.title = exercise_display
        self.angle_keys = angle_keys
        self.side_w = side_w
        self.strip_h = strip_h
        self.target_h = target_h
        self.strip = SignalStrip()
        self.last_rep: dict | None = None
        self.cue: tuple[str, float] | None = None
        self.yaw = -60.0

    def note_rep(self, t: float, rep) -> None:
        self.strip.mark(t, rep.index, rep.score)
        top = rep.faults[0] if rep.faults else None
        self.last_rep = {"index": rep.index, "score": rep.score, "fault": top.fault if top else None,
                         "side": top.side if top else None}

    def note_cue(self, t: float, cue: str) -> None:
        if not cue.isdigit():
            self.cue = (cue, t)

    def render(self, frame: np.ndarray, kp2d: np.ndarray | None, upd, t: float, fps: float | None = None,
               status: str = "") -> np.ndarray:
        h0, w0 = frame.shape[:2]
        sc = self.target_h / h0
        img = cv2.resize(frame, (int(round(w0 * sc)), self.target_h), interpolation=cv2.INTER_AREA)
        bad2d, bad3d = _bad_sets(upd.live if upd else [])
        if kp2d is not None:
            k = kp2d.copy()
            k[:, :2] *= sc
            draw_skeleton_2d(img, k, bad2d, scale=self.target_h / 720)
        H, W = img.shape[:2]
        canvas = np.full((H + self.strip_h, W + self.side_w, 3), BG, np.uint8)
        canvas[:H, :W] = img
        self._hud(canvas, upd, fps, status)
        side = canvas[:H, W:]
        view_h = int(H * 0.55)
        draw_skeleton_3d(side[:view_h], upd.pose3d if upd else None, yaw=self.yaw, bad=bad3d,
                         up=upd.up if upd else None)
        self._angles(side[view_h:], upd)
        self.strip.add(t, upd.signal if upd else None, upd.phase if upd else "top")
        self.strip.draw(canvas[H:, :], f"rep signal: {upd.signal_name if upd else ''}")
        self._cue(canvas, W, H, t)
        return canvas

    def _hud(self, c: np.ndarray, upd, fps, status) -> None:
        ov = c.copy()
        cv2.rectangle(ov, (12, 12), (330, 158), BG, -1)
        cv2.addWeighted(ov, 0.72, c, 0.28, 0, c)
        text(c, self.title.upper(), (24, 20), 0.55, MUTED)
        count = upd.count if upd else 0
        text(c, f"{count}", (24, 44), 2.2, FG, 3)
        text(c, "reps", (24 + 40 * len(str(count)) + 12, 82), 0.6, MUTED)
        phase = upd.phase if upd else "top"
        pc = PHASE_COL.get(phase, MUTED)
        label = {"top": "LOCKOUT", "down": "DOWN", "bottom": "BOTTOM", "up": "UP"}.get(phase, phase.upper())
        cv2.rectangle(c, (200, 50), (318, 78), pc, -1)
        text(c, label, (259, 64), 0.5, BG, 1, anchor="cc")
        if self.last_rep and self.last_rep.get("score") is not None:
            s = self.last_rep["score"]
            col = GOOD if s >= 80 else WARN if s >= 60 else BAD
            text(c, f"rep {self.last_rep['index']}: {s:.0f}/100", (24, 112), 0.55, col)
            f = self.last_rep.get("fault")
            if f:
                side = f" ({self.last_rep['side']})" if self.last_rep.get("side") else ""
                text(c, f"{f}{side}", (24, 136), 0.48, col)
            else:
                text(c, "clean rep", (24, 136), 0.48, GOOD)
        info = " | ".join(x for x in [f"{fps:.0f} fps" if fps else "", status] if x)
        if info:
            text(c, info, (c.shape[1] - self.side_w - 12, 14), 0.45, MUTED, anchor="tr")

    def _angles(self, panel: np.ndarray, upd) -> None:
        panel[:] = BG
        text(panel, "joint angles", (12, 10), 0.45, MUTED)
        m = upd.metrics if upd else {}
        live = {v["id"] for v in (upd.live if upd else [])}
        y = 36
        for key, label in self.angle_keys:
            v = m.get(key)
            s = "--" if v is None else (f"{v:.0f} deg" if abs(v) >= 3 or "ratio" not in key else f"{v:.2f}")
            if v is not None and key in ("knee_ankle_ratio", "hip_depth", "stance_ratio"):
                s = f"{v:.2f}"
            elif v is not None and key.startswith("knee_medial"):
                s = f"{v:.1f} %leg"
            col = BAD if any(key.startswith(i) for i in live) else FG
            text(panel, label, (12, y), 0.48, MUTED)
            text(panel, s, (panel.shape[1] - 14, y), 0.55, col, anchor="tr")
            y += 26
            if y > panel.shape[0] - 10:
                break
        if upd and upd.live:
            text(panel, upd.live[0]["fault"], (12, panel.shape[0] - 30), 0.5, BAD)

    def _cue(self, c: np.ndarray, W: int, H: int, t: float) -> None:
        if not self.cue or t - self.cue[1] > 3.0:
            return
        msg = self.cue[0]
        (tw, th), _ = cv2.getTextSize(msg, FONT, 0.85, 2)
        x0, y0 = (W - tw) // 2 - 18, H - 70
        ov = c.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + tw + 36, y0 + th + 26), BAD, -1)
        cv2.addWeighted(ov, 0.85, c, 0.15, 0, c)
        text(c, msg, (x0 + 18, y0 + 12), 0.85, FG, 2)


def angle_keys_for(exercise) -> list[tuple[str, str]]:
    """The readouts worth showing for an exercise: its rep signal and the metrics its rules check."""
    labels = {"knee_flexion": "knee flexion", "knee_flexion_l": "knee L", "knee_flexion_r": "knee R",
              "knee_flexion_max": "front knee", "hip_flexion": "hip flexion", "elbow_flexion": "elbow flexion",
              "elbow_flexion_max": "elbow flexion", "arm_elevation": "arm elevation",
              "arm_elevation_max": "arm elevation", "trunk_inclination": "trunk lean", "spine_flexion": "spine bend",
              "knee_ankle_ratio": "knee/ankle width", "knee_medial_max": "knee drift in", "hip_depth": "hip height",
              "body_line": "body line", "stance_ratio": "stance width", "knee_flexion_asym": "L/R knee diff",
              "elbow_flexion_asym": "L/R elbow diff", "hip_flexion_asym": "L/R hip diff",
              "arm_elevation_asym": "L/R arm diff"}
    keys = [] if exercise.auto else [exercise.signal]
    if exercise.auto:
        keys = ["knee_flexion", "hip_flexion", "elbow_flexion", "arm_elevation", "trunk_inclination"]
    for r in exercise.rules:
        for k in (r.metric, r.a, r.b):
            if k and k not in keys:
                keys.append(k)
    if exercise.signal in ("knee_flexion", "knee_flexion_max"):
        keys[1:1] = [k for k in ("knee_flexion_l", "knee_flexion_r") if k not in keys]
    return [(k, labels.get(k, k.replace("_", " "))) for k in keys][:9]
