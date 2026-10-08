"""FormFlow web app: FastAPI + a single-page UI.

    GET  /                       the app (static/)
    GET  /api/health             liveness
    GET  /api/status             device, models, settings
    GET  /api/exercises          exercise list with rules and camera advice
    WS   /ws/live                live coaching: the browser sends JPEG frames (binary) and JSON control
                                 messages; every frame gets a JSON reply (skeleton, angles, phase, reps, cues)
    POST /api/analyze            upload a video (+ exercise) -> background job
    GET  /api/jobs/{id}          job progress / result
    GET  /api/sessions           analysed videos and finished live sets
    GET  /sessions/...           their files (annotated.mp4, report.html, ...)
    GET  /samples/...            demo videos shipped in the repo

One model set (detector, pose, lifter) is shared; every live connection gets its own tracker, filter,
streaming lifter and coach. Frames are processed one at a time per connection: the browser sends the
next frame only after the reply, so a slow machine lowers the frame rate instead of building a backlog.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..coach.exercises import exercises, get_exercise
from ..paths import SAMPLES_DIR, sessions_dir

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 1024 * 1024 * 1024  # 1 GB


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


class State:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.engine = None
        self.engine_error: str | None = None
        self.lock = threading.Lock()  # one GPU: serialise model calls between connections / jobs
        self.jobs: dict[str, dict] = {}
        self._loading = threading.Event()

    def ensure_engine(self):
        if self.engine is not None:
            return self.engine
        with self.lock:
            if self.engine is None:
                from ..engine import build_engine

                try:
                    self.engine = build_engine(self.cfg, "live")
                    self.engine_error = None
                except Exception as e:  # surfaced in /api/status and the UI
                    self.engine_error = f"{type(e).__name__}: {e}"
                    raise
        return self.engine


def create_app(cfg: dict, engine=None) -> FastAPI:
    """``engine`` injects a ready model set (tests); otherwise it is built on first use."""
    app = FastAPI(title="FormFlow", version=__version__)
    st = State(cfg)
    st.engine = engine
    app.state.formflow = st

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/api/status")
    def status():
        from ..data.download import MODELS, have_model
        from ..engine import pick_models

        return {"version": __version__, "models": pick_models(cfg, "live"), "loaded": st.engine is not None,
                "error": st.engine_error, "have": {k: have_model(k) for k in MODELS},
                "live": cfg.get("live", {}), "live_cpu": cfg.get("live_cpu", {}), "engine": st.engine.describe() if st.engine else None}

    @app.get("/api/exercises")
    def list_exercises():
        return [ex.to_dict() for ex in exercises().values()]

    @app.get("/api/samples")
    def samples():
        if not SAMPLES_DIR.is_dir():
            return []
        out = []
        for p in sorted(SAMPLES_DIR.glob("*.json")):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if (SAMPLES_DIR / meta.get("video", "")).is_file():
                out.append(meta)
        return out

    # ------------------------------------------------------------------ live
    @app.websocket("/ws/live")
    async def live(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        try:
            engine = await loop.run_in_executor(None, st.ensure_engine)
        except Exception as e:
            await ws.send_json({"type": "error", "message": f"models could not be loaded: {e}"})
            await ws.close()
            return
        from ..config import live_settings
        from ..pipeline import LivePipeline

        lv = live_settings(cfg, engine.device)
        ex = get_exercise("squat")
        pipe = None
        t_start = time.perf_counter()
        frames = 0
        last_t = None
        fps_ema = None
        await ws.send_json({"type": "ready", "engine": engine.describe()})
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "exercise":
                        ex = get_exercise(data.get("name", "squat"))
                        if pipe is not None:
                            pipe.set_exercise(ex)
                        await ws.send_json({"type": "exercise", "exercise": ex.to_dict()})
                    elif kind == "reset":
                        pipe = None
                        await ws.send_json({"type": "reset"})
                    elif kind == "finish":
                        summary = pipe.coach.summary() if pipe else None
                        if summary and summary["reps"]:
                            sid = f"{time.strftime('%Y%m%d-%H%M%S')}_live_{ex.name}"
                            d = sessions_dir() / sid
                            d.mkdir(parents=True, exist_ok=True)
                            (d / "report.json").write_text(json.dumps({"mode": "live", "exercise": ex.name,
                                                                       "display": ex.display, "summary": summary},
                                                                      indent=1, default=_jsonable), encoding="utf-8")
                        await ws.send_text(json.dumps({"type": "summary", "summary": summary}, default=_jsonable))
                        pipe = None
                    continue
                buf = msg.get("bytes")
                if not buf:
                    continue
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if pipe is None:
                    pipe = LivePipeline(engine.tracker(int(lv.get("det_every", 10)), async_detect=True), engine.lifter,
                                        ex, (img.shape[1], img.shape[0]), fps=30.0, window=int(lv.get("window", 81)),
                                        stride=int(lv.get("stride", 2)), one_euro=cfg.get("one_euro"),
                                        one_euro_3d=lv.get("one_euro_3d"), filter_2d_for_lift=bool(lv.get("one_euro_2d_lift", False)), up_mode=cfg.get("up", "auto"))
                    engine.lifter.flip = bool(lv.get("flip", False))
                    engine.lifter.half = bool(lv.get("half", True)) and engine.lifter.device == "cuda"
                    t_start = time.perf_counter()
                t = time.perf_counter() - t_start

                def work(pipe=pipe, img=img, t=t):
                    with st.lock:
                        return pipe.process(img, t)

                res = await loop.run_in_executor(None, work)
                frames += 1
                if last_t is not None:
                    inst = 1.0 / max(t - last_t, 1e-3)
                    fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
                last_t = t
                await ws.send_text(json.dumps(frame_message(res, img.shape, fps_ema), default=_jsonable))
        except WebSocketDisconnect:
            pass

    # ------------------------------------------------------------------ video analysis jobs
    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...), exercise: str = Form("auto")):
        try:
            ex = get_exercise(exercise)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e
        jid = uuid.uuid4().hex[:10]
        name = Path(file.filename or "video.mp4").name
        out = sessions_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(name).stem[:40]}_{jid}"
        out.mkdir(parents=True, exist_ok=True)
        dest = out / ("input" + (Path(name).suffix.lower() or ".mp4"))
        size = 0
        with dest.open("wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    f.close()
                    shutil.rmtree(out, ignore_errors=True)
                    raise HTTPException(413, "video too large (1 GB max)")
                f.write(chunk)
        job = {"id": jid, "state": "queued", "stage": "queued", "progress": 0.0, "exercise": ex.name,
               "dir": out.name, "created": time.time()}
        st.jobs[jid] = job
        threading.Thread(target=_run_job, args=(st, job, dest, ex.name, out), daemon=True).start()
        return job

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        if jid not in st.jobs:
            raise HTTPException(404, "no such job")
        return st.jobs[jid]

    @app.get("/api/sessions")
    def sessions():
        root = sessions_dir()
        out = []
        if root.is_dir():
            for d in sorted(root.iterdir(), reverse=True):
                rj = d / "report.json"
                if not rj.is_file():
                    continue
                try:
                    r = json.loads(rj.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                s = r.get("summary", {})
                out.append({"id": d.name, "mode": r.get("mode"), "exercise": r.get("display", r.get("exercise")),
                            "reps": s.get("reps"), "score": s.get("mean_score"),
                            "faults": [f["fault"] for f in s.get("faults", [])][:3],
                            "video": f"/sessions/{d.name}/annotated.mp4" if (d / "annotated.mp4").is_file() else None,
                            "report": f"/sessions/{d.name}/report.html" if (d / "report.html").is_file() else None})
        return out[:100]

    @app.get("/sessions/{sid}/{name}")
    def session_file(sid: str, name: str):
        root = sessions_dir().resolve()
        p = (root / sid / name).resolve()
        if root not in p.parents or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    if SAMPLES_DIR.is_dir():
        app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

    @app.exception_handler(KeyError)
    async def key_error(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return app


def frame_message(res, shape, fps) -> dict:
    """Compact per-frame reply. 2D keypoints are normalised to 0..1 so the browser can draw at any size."""
    h, w = shape[:2]
    u = res.update
    kp = res.kp2d.copy()
    kp[:, 0] /= w
    kp[:, 1] /= h
    msg = {
        "type": "frame", "frame": res.frame, "visible": res.visible,
        "kp2d": np.round(kp, 4).tolist() if res.visible else None,
        "pose3d": None if u.pose3d is None else np.round((u.pose3d - u.pose3d[:1]) / max(float(np.ptp(u.pose3d[:, 1])), 1e-6), 4).tolist(),
        "up": None if u.up is None else np.round(u.up, 4).tolist(),
        "phase": u.phase, "count": u.count, "excursion": round(float(u.excursion), 3),
        "signal": None if u.signal is None else round(float(u.signal), 2), "signal_name": u.signal_name,
        "metrics": {k: round(float(v), 2) for k, v in u.metrics.items()},
        "live": u.live, "cues": u.cues,
        "reps": [r.to_dict() | {"faults": [f.to_dict() for f in r.faults]} for r in u.reps],
        "timings": {k: round(v, 1) for k, v in res.timings.items()}, "fps": None if fps is None else round(fps, 1),
    }
    return msg


def _run_job(st: State, job: dict, video: Path, exercise: str, out: Path) -> None:
    from ..analyze import analyze_video
    from ..engine import build_engine

    def progress(stage: str, frac: float) -> None:
        job.update(stage=stage, progress=round(frac, 3))

    try:
        job["state"] = "running"
        with st.lock:  # a whole video at a time: keeps live sessions and jobs off each other's GPU memory
            if st.engine is None:
                st.engine = build_engine(st.cfg, "live")
            eng = st.engine
            live_flip, live_half = eng.lifter.flip, eng.lifter.half
            try:
                r = analyze_video(video, exercise, st.cfg, engine=eng, out_dir=out, progress=progress)
            finally:
                eng.lifter.flip, eng.lifter.half = live_flip, live_half
        s = r["summary"]
        job.update(state="done", progress=1.0, stage="done", reps=s["reps"], score=s["mean_score"],
                   faults=s["faults"], report=f"/sessions/{out.name}/report.html",
                   video=f"/sessions/{out.name}/annotated.mp4")
    except Exception as e:
        job.update(state="failed", error=f"{type(e).__name__}: {e}")


def _self_signed(dir_: Path) -> tuple[str, str]:
    """A self-signed certificate for --https (browsers allow the camera only on https or localhost)."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    cert, key = dir_ / "formflow-cert.pem", dir_ / "formflow-key.pem"
    if cert.is_file() and key.is_file():
        return str(cert), str(key)
    dir_.mkdir(parents=True, exist_ok=True)
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FormFlow local")])
    alt = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    with contextlib.suppress(ValueError):
        alt.append(x509.IPAddress(ipaddress.ip_address(_lan_ip() or "127.0.0.1")))
    now = datetime.datetime.now(datetime.timezone.utc)
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
         .not_valid_after(now + datetime.timedelta(days=825)).add_extension(x509.SubjectAlternativeName(alt), False)
         .sign(k, hashes.SHA256()))
    cert.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    return str(cert), str(key)


def _lan_ip() -> str | None:
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def serve(cfg: dict, host: str | None = None, port: int | None = None, open_browser: bool = True,
          https: bool = False) -> int:
    import uvicorn

    from ..paths import home

    sc = cfg.get("server", {})
    host = host or sc.get("host", "127.0.0.1")
    port = int(port or sc.get("port", 8780))
    app = create_app(cfg)
    scheme = "https" if https else "http"
    url = f"{scheme}://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}"
    ssl = {}
    if https:
        cert, key = _self_signed(home() / "certs")
        ssl = {"ssl_certfile": cert, "ssl_keyfile": key}
    print(f"FormFlow {__version__} on {url}")
    if host == "0.0.0.0":
        ip = _lan_ip()
        if ip:
            print(f"On your phone (same Wi-Fi): {scheme}://{ip}:{port}"
                  + ("   (accept the certificate warning once)" if https else "   (camera needs --https)"))

    ready_file = os.environ.get("FORMFLOW_READY_FILE")

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", **ssl)
    server = uvicorn.Server(config)

    def _after_start():
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        if ready_file:
            Path(ready_file).write_text("ok", encoding="utf-8")
        if open_browser:
            webbrowser.open(url)
        # Warm the models up in the background so the first live frame doesn't wait for them.
        try:
            app.state.formflow.ensure_engine()
        except Exception as e:
            print(f"warning: models not loaded yet ({e}); run `formflow models download`")

    threading.Thread(target=_after_start, daemon=True).start()
    server.run()
    return 0
