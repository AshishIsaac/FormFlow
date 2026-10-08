"""``formflow live``: real-time coaching in a desktop window (OpenCV), from a webcam or a video file.

A reader thread always holds the newest camera frame, so a slow machine drops frames instead of
falling behind. A video file is paced at its own frame rate, as if it were a camera.

Keys:  q / Esc quit   r reset the set   n next exercise   f finish set (summary)   m mute   space pause
Spoken cues use the operating system's own voice (Windows SAPI, macOS `say`, Linux `espeak`).
"""

from __future__ import annotations

import contextlib
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .coach.exercises import exercises, get_exercise
from .pipeline import LivePipeline
from .render.overlay import Renderer, angle_keys_for


class Speaker:
    """Speaks on a background thread; drops queued phrases so it never lags behind the athlete."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.q: queue.Queue[str] = queue.Queue()
        self.cmd = self._find()
        if self.cmd is None:
            self.enabled = False
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _find():
        if sys.platform == "win32":
            return ["powershell", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($args[0])"]
        if sys.platform == "darwin" and shutil.which("say"):
            return ["say"]
        for exe in ("espeak-ng", "espeak", "spd-say"):
            if shutil.which(exe):
                return [exe]
        return None

    def say(self, text: str) -> None:
        if self.enabled:
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.q.put(text)

    def _run(self) -> None:
        while True:
            text = self.q.get()
            if self.cmd is None:
                continue
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run([*self.cmd, text], capture_output=True, timeout=15)


class FrameSource:
    """Newest-frame reader for a camera index or a video file (paced to real time)."""

    def __init__(self, source, mirror: bool) -> None:
        self.is_file = isinstance(source, str)
        self.cap = cv2.VideoCapture(source if self.is_file else int(source))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open {'video' if self.is_file else 'camera'} {source}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.mirror = mirror
        self.frame: np.ndarray | None = None
        self.ended = False
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        idx = 0
        while not self.ended:
            if self.is_file:
                due = self._t0 + idx / self.fps
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            ok, img = self.cap.read()
            if not ok:
                self.ended = True
                break
            idx += 1
            if self.mirror:
                img = cv2.flip(img, 1)
            with self._lock:
                self.frame = img

    def read(self) -> np.ndarray | None:
        with self._lock:
            img, self.frame = self.frame, None
        return img


def run_desktop(cfg: dict, source=0, exercise: str = "squat", speak: bool = True, record: str | None = None,
                mirror: bool = True) -> int:
    from .analyze import VideoWriter
    from .engine import build_engine

    print("Loading models ...", flush=True)
    engine = build_engine(cfg, "live")
    from .config import live_settings

    lv = live_settings(cfg, engine.device)
    engine.lifter.flip = bool(lv.get("flip", False))
    engine.lifter.half = bool(lv.get("half", True)) and engine.lifter.device == "cuda"
    print(f"  {engine.describe()}")
    src = FrameSource(source, mirror)
    names = list(exercises())
    ex = get_exercise(exercise)
    speaker = Speaker(speak)
    writer = None
    win = "FormFlow"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    pipe = renderer = None
    t0 = time.perf_counter()
    fps_ema = None
    last = time.perf_counter()
    paused = False
    max_side = int(lv.get("max_side", 960))

    def new_pipe(img):
        nonlocal renderer
        renderer = Renderer(ex.display, angle_keys_for(ex))
        return LivePipeline(engine.tracker(int(lv.get("det_every", 10)), async_detect=True), engine.lifter, ex,
                            (img.shape[1], img.shape[0]), fps=src.fps if src.is_file else 30.0,
                            window=int(lv.get("window", 81)), stride=int(lv.get("stride", 2)),
                            one_euro=cfg.get("one_euro"), one_euro_3d=lv.get("one_euro_3d"), filter_2d_for_lift=bool(lv.get("one_euro_2d_lift", False)),
                            up_mode=cfg.get("up", "auto"))

    try:
        while True:
            img = None if paused else src.read()
            if img is None:
                if src.ended and not paused:
                    break
                key = cv2.waitKey(5) & 0xFF
            else:
                if max(img.shape[:2]) > max_side:
                    s = max_side / max(img.shape[:2])
                    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
                if pipe is None:
                    pipe = new_pipe(img)
                t = time.perf_counter() - t0
                res = pipe.process(img, t)
                for rep in res.update.reps:
                    renderer.note_rep(t, rep)
                for c in res.update.cues:
                    renderer.note_cue(t, c)
                    speaker.say(c)
                now = time.perf_counter()
                fps_ema = 1 / max(now - last, 1e-3) if fps_ema is None else 0.9 * fps_ema + 0.1 / max(now - last, 1e-3)
                last = now
                out = renderer.render(img, res.kp2d if res.visible else None, res.update, t, fps_ema,
                                      status=f"{engine.device} - q quit, n exercise, r reset, f finish")
                cv2.imshow(win, out)
                if record:
                    if writer is None:
                        writer = VideoWriter(Path(record), 30.0, (out.shape[1] - out.shape[1] % 2, out.shape[0] - out.shape[0] % 2))
                    writer.write(out)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") or key == ord("n"):
                if key == ord("n"):
                    ex = get_exercise(names[(names.index(ex.name) + 1) % len(names)])
                    speaker.say(ex.display)
                pipe = None
            elif key == ord("f") and pipe is not None:
                s = pipe.coach.summary()
                msg = f"Set complete. {s['reps']} reps." + (f" Work on: {s['faults'][0]['cue']}." if s["faults"] else " Great form.")
                print(msg)
                speaker.say(msg)
                pipe = None
            elif key == ord("m"):
                speaker.enabled = not speaker.enabled and speaker.cmd is not None
            elif key == ord(" "):
                paused = not paused
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        src.ended = True
        if writer is not None:
            writer.close()
        cv2.destroyAllWindows()
    if pipe is not None:
        s = pipe.coach.summary()
        print(f"{s['display']}: {s['reps']} reps, mean score {s['mean_score']}")
    return 0
