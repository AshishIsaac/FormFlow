// FormFlow browser app: camera -> server (WebSocket, JPEG frames) -> skeleton, angles, reps, cues.
"use strict";

const $ = (id) => document.getElementById(id);
const COCO_BONES = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]];
const COCO_SIDE = [0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1];
const H36_BONES = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]];
const H36_SIDE = [0, -1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1];
const H36_INDEX = { pelvis: 0, r_hip: 1, r_knee: 2, r_ankle: 3, l_hip: 4, l_knee: 5, l_ankle: 6, spine: 7, thorax: 8, neck: 9, head: 10, l_shoulder: 11, l_elbow: 12, l_wrist: 13, r_shoulder: 14, r_elbow: 15, r_wrist: 16 };
const H36_TO_COCO = { l_knee: [13], r_knee: [14], l_hip: [11], r_hip: [12], l_ankle: [15], r_ankle: [16], l_elbow: [7], r_elbow: [8], l_wrist: [9], r_wrist: [10], l_shoulder: [5], r_shoulder: [6], pelvis: [11, 12], spine: [5, 6, 11, 12], thorax: [5, 6], neck: [0], head: [0, 1, 2] };
const LABELS = { knee_flexion: "knee flexion", knee_flexion_l: "knee L", knee_flexion_r: "knee R", knee_flexion_max: "front knee", hip_flexion: "hip flexion", elbow_flexion: "elbow flexion", elbow_flexion_max: "elbow flexion", arm_elevation: "arm elevation", arm_elevation_max: "arm elevation", trunk_inclination: "trunk lean", spine_flexion: "spine bend", knee_ankle_ratio: "knee / ankle width", knee_medial_max: "knee drift in", hip_depth: "hip height", body_line: "body line", stance_ratio: "stance width", knee_flexion_asym: "L/R knee diff", elbow_flexion_asym: "L/R elbow diff", hip_flexion_asym: "L/R hip diff", arm_elevation_asym: "L/R arm diff" };

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const state = {
  exercises: [], exercise: null, ws: null, running: false, inflight: false, stream: null, last: null,
  signal: [], repMarks: [], yaw: -60, dragging: false, cueTimer: null, mirror: false, t0: 0, speak: true,
};

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "sessions") loadSessions();
  if (b.dataset.tab === "analyze") loadSamples();
}));

// ------------------------------------------------------------------ status + exercises
async function init() {
  try {
    const st = await (await fetch("/api/status")).json();
    setStatus(st.error ? "err" : "ok", st.error ? "models missing" : `${st.models.device.toUpperCase()} - ${st.models.lifter}`);
  } catch (e) { setStatus("err", "server offline"); }
  state.exercises = await (await fetch("/api/exercises")).json();
  const order = ["squat", "deadlift", "push_up", "lunge", "bicep_curl", "overhead_press", "pull_up", "jumping_jack", "situp", "bench_press", "front_raise", "auto"];
  state.exercises.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
  state.ruleMetrics = Object.fromEntries(state.exercises.map((e) => [e.name, e.metrics || []]));
  for (const sel of [$("exercise"), $("exercise2")]) {
    sel.innerHTML = state.exercises.map((e) => `<option value="${e.name}">${e.display}</option>`).join("");
  }
  $("exercise2").value = "auto";
  selectExercise("squat");
  try {
    const samples = await (await fetch("/api/samples")).json();
    for (const s of samples) {
      const o = document.createElement("option");
      o.value = "sample:" + s.video; o.textContent = "Sample: " + s.title; o.dataset.exercise = s.exercise;
      $("source").appendChild(o);
    }
  } catch (e) { /* no samples shipped */ }
}

function setStatus(kind, text) {
  $("status").className = "status " + kind;
  $("statusText").textContent = text;
}

function selectExercise(name) {
  state.exercise = state.exercises.find((e) => e.name === name) || state.exercises[0];
  $("exercise").value = state.exercise.name;
  $("adviceText").textContent = "Camera: " + (state.exercise.camera || "whole body in frame");
  $("cameraHint").textContent = state.exercise.rules.length
    ? "Checks: " + state.exercise.rules.map((r) => r.fault.toLowerCase()).join(" - ")
    : "Counts reps of any repetitive movement.";
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify({ type: "exercise", name: state.exercise.name }));
  renderAngles({});
}
$("exercise").addEventListener("change", (e) => selectExercise(e.target.value));
$("speak").addEventListener("change", (e) => { state.speak = e.target.checked; if (!state.speak) speechSynthesis.cancel(); });
$("source").addEventListener("change", (e) => {
  const o = e.target.selectedOptions[0];
  if (o && o.dataset.exercise) selectExercise(o.dataset.exercise);
});

// ------------------------------------------------------------------ live session
$("startBtn").addEventListener("click", () => (state.running ? stop() : start()));
$("finishBtn").addEventListener("click", finishSet);

async function start() {
  const src = $("source").value;
  const video = $("video");
  try {
    if (src.startsWith("sample:")) {
      video.srcObject = null;
      video.src = "/samples/" + src.slice(7);
      video.loop = false;
      state.mirror = false;
      video.onended = () => finishSet();
    } else {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        $("cameraHint").textContent = "This browser blocks the camera here. Use http://localhost, or start the server with --https for phones.";
        return;
      }
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: src, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.src = "";
      video.srcObject = state.stream;
      state.mirror = src === "user";
    }
    await video.play();
  } catch (e) {
    $("cameraHint").textContent = "Could not start the camera: " + e.message;
    return;
  }
  $("stage").classList.toggle("mirror", state.mirror);
  $("placeholder").hidden = true;
  resetSet();
  connect();
  state.running = true;
  $("startBtn").textContent = "Stop";
  $("finishBtn").disabled = false;
}

function stop() {
  state.running = false;
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  $("video").pause();
  if (state.ws) state.ws.close();
  state.ws = null;
  $("startBtn").textContent = "Start";
  $("finishBtn").disabled = true;
  $("placeholder").hidden = false;
  $("perf").textContent = "";
}

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/live`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  state.inflight = false;
  ws.onopen = () => setStatus("", "loading models...");
  ws.onmessage = (ev) => onMessage(JSON.parse(ev.data));
  ws.onclose = () => { if (state.running) setStatus("err", "disconnected"); };
}

function onMessage(m) {
  if (m.type === "ready") {
    setStatus("ok", `${m.engine.device.toUpperCase()} - live`);
    state.ws.send(JSON.stringify({ type: "exercise", name: state.exercise.name }));
    pump();
  } else if (m.type === "frame") {
    state.inflight = false;
    onFrame(m);
    requestAnimationFrame(pump);
  } else if (m.type === "summary") {
    showSummary(m.summary);
  } else if (m.type === "error") {
    setStatus("err", "error");
    $("cameraHint").textContent = m.message;
    stop();
  }
}

const grab = document.createElement("canvas");
function pump() {
  if (!state.running || state.inflight || !state.ws || state.ws.readyState !== 1) return;
  const v = $("video");
  if (!v.videoWidth || v.paused || v.ended) { setTimeout(pump, 50); return; }
  const scale = Math.min(1, 640 / Math.max(v.videoWidth, v.videoHeight));
  grab.width = Math.round(v.videoWidth * scale);
  grab.height = Math.round(v.videoHeight * scale);
  grab.getContext("2d").drawImage(v, 0, 0, grab.width, grab.height);
  state.inflight = true;
  grab.toBlob(async (b) => {
    if (!b || !state.ws || state.ws.readyState !== 1) { state.inflight = false; return; }
    state.ws.send(await b.arrayBuffer());
  }, "image/jpeg", 0.75);
}

function resetSet() {
  state.signal = []; state.repMarks = []; state.t0 = performance.now();
  $("repList").innerHTML = ""; $("count").textContent = "0";
  $("lastScore").textContent = "-"; $("lastScore").className = "score";
  $("lastFault").textContent = "Faults and cues appear here after each rep.";
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify({ type: "reset" }));
}

function finishSet() {
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify({ type: "finish" }));
}

function showSummary(s) {
  if (!s) return;
  const faults = s.faults.length ? s.faults.map((f) => `${f.fault} (${f.reps}/${s.reps})`).join(", ") : "no form faults";
  $("lastFault").innerHTML = `<b>Set done: ${s.reps} reps, mean score ${s.mean_score ?? "-"}</b><br>${esc(faults)}`;
  say(`Set complete. ${s.reps} reps.` + (s.faults[0] ? ` Work on: ${s.faults[0].cue}.` : " Great form."));
  state.signal = []; state.repMarks = [];
  $("count").textContent = "0"; $("repList").innerHTML = "";
}

// ------------------------------------------------------------------ per-frame UI
function onFrame(m) {
  state.last = m;
  const t = (performance.now() - state.t0) / 1000;
  drawOverlay(m);
  $("count").textContent = m.count;
  const ph = $("phase");
  ph.className = "phase " + m.phase;
  ph.textContent = { top: "LOCKOUT", down: "DOWN", bottom: "BOTTOM", up: "UP" }[m.phase] || m.phase.toUpperCase();
  if (m.signal !== null) state.signal.push([t, m.signal, m.phase]);
  while (state.signal.length && t - state.signal[0][0] > 15) state.signal.shift();
  $("signalName").textContent = (m.signal_name || "").replaceAll("_", " ");
  for (const r of m.reps) addRep(r, t);
  for (const c of m.cues) cue(c);
  renderAngles(m.metrics, m.live);
  drawSignal(t);
  draw3d(m.pose3d, m.up, m.live);
  const tm = m.timings || {};
  $("perf").textContent = `${m.fps ? m.fps.toFixed(0) + " fps - " : ""}2D ${(tm.pose2d_ms || 0).toFixed(0)} ms - 3D ${(tm.lift_ms || 0).toFixed(0)} ms`;
}

function addRep(r, t) {
  state.repMarks.push([t, r.index, r.score]);
  const cls = r.score >= 80 ? "good" : r.score >= 60 ? "warn" : "bad";
  $("lastScore").textContent = `${Math.round(r.score)}/100`;
  $("lastScore").className = "score " + cls;
  const f = r.faults[0];
  $("lastFault").innerHTML = f ? `<b>${esc(f.fault)}${f.side ? " (" + f.side + ")" : ""}</b><br><span class="muted">${esc(f.cue)}</span>` : "Clean rep.";
  const li = document.createElement("li");
  li.innerHTML = `<span class="pill ${cls}">${Math.round(r.score)}</span>${f ? esc(f.fault) : "clean"} <span class="muted">${r.duration_s.toFixed(1)}s</span>`;
  $("repList").prepend(li);
}

function cue(text) {
  say(text);
  if (/^\d+$/.test(text)) return;
  const el = $("cue");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(state.cueTimer);
  state.cueTimer = setTimeout(() => (el.hidden = true), 3000);
}

function say(text) {
  if (!state.speak || !("speechSynthesis" in window)) return;
  if (speechSynthesis.pending) speechSynthesis.cancel(); // never fall behind the athlete
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.1;
  speechSynthesis.speak(u);
}

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ------------------------------------------------------------------ drawing
function fitCanvas(c) {
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return [w, h, dpr];
}

function badSets(live) {
  const coco = new Set(), h36 = new Set();
  for (const v of live || []) for (const j of v.joints || []) {
    (H36_TO_COCO[j] || []).forEach((k) => coco.add(k));
    if (j in H36_INDEX) h36.add(H36_INDEX[j]);
  }
  return [coco, h36];
}

function drawOverlay(m) {
  const c = $("overlay"), v = $("video");
  const [w, h, dpr] = fitCanvas(c);
  const g = c.getContext("2d");
  g.clearRect(0, 0, w, h);
  if (!m.kp2d || !v.videoWidth) return;
  // the video is letterboxed with object-fit: contain
  const s = Math.min(w / v.videoWidth, h / v.videoHeight);
  const vw = v.videoWidth * s, vh = v.videoHeight * s, ox = (w - vw) / 2, oy = (h - vh) / 2;
  const P = (k) => [ox + k[0] * vw, oy + k[1] * vh];
  const [bad] = badSets(m.live);
  const L = css("--left"), R = css("--right"), B = css("--bad");
  g.lineCap = "round";
  g.lineWidth = 4 * dpr;
  for (const [a, b] of COCO_BONES) {
    const ka = m.kp2d[a], kb = m.kp2d[b];
    if (ka[2] < 0.3 || kb[2] < 0.3) continue;
    g.strokeStyle = bad.has(a) && bad.has(b) ? B : COCO_SIDE[b] > 0 ? L : COCO_SIDE[b] < 0 ? R : "#ddd";
    const [x1, y1] = P(ka), [x2, y2] = P(kb);
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
  }
  for (let j = 0; j < 17; j++) {
    const k = m.kp2d[j];
    if (k[2] < 0.3) continue;
    const [x, y] = P(k);
    g.fillStyle = bad.has(j) ? B : "#fff";
    g.beginPath(); g.arc(x, y, (bad.has(j) ? 7 : 4) * dpr, 0, 6.283); g.fill();
  }
}

function renderAngles(metrics, live) {
  const ex = state.exercise;
  if (!ex) return;
  let keys = ex.signal === "auto" ? ["knee_flexion", "hip_flexion", "elbow_flexion", "arm_elevation", "trunk_inclination"] : [ex.signal];
  if (ex.signal.startsWith("knee_flexion")) keys.push("knee_flexion_l", "knee_flexion_r");
  for (const k of (state.ruleMetrics || {})[ex.name] || []) if (!keys.includes(k)) keys.push(k);
  $("angles").innerHTML = keys.slice(0, 9).map((k) => {
    const v = metrics[k];
    let s = "--";
    if (v !== undefined) {
      if (["knee_ankle_ratio", "hip_depth", "stance_ratio"].includes(k)) s = v.toFixed(2);
      else if (k.startsWith("knee_medial")) s = v.toFixed(1) + " %";
      else s = Math.round(v) + "°";
    }
    return `<dt>${LABELS[k] || k.replaceAll("_", " ")}</dt><dd class="${(live || []).some((x) => x.metric === k) ? "bad" : ""}">${s}</dd>`;
  }).join("");
}

function drawSignal(tNow) {
  const c = $("signal");
  const [w, h, dpr] = fitCanvas(c);
  const g = c.getContext("2d");
  g.clearRect(0, 0, w, h);
  const d = state.signal;
  if (d.length < 2) return;
  let lo = Infinity, hi = -Infinity;
  for (const p of d) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }
  if (hi - lo < 10) { const mid = (hi + lo) / 2; lo = mid - 5; hi = mid + 5; }
  const X = (t) => w - 6 * dpr - ((tNow - t) / 15) * (w - 12 * dpr);
  const Y = (v) => h - 6 * dpr - ((v - lo) / (hi - lo)) * (h - 26 * dpr);
  const col = { top: css("--muted"), down: css("--left"), bottom: css("--right"), up: css("--good") };
  g.lineWidth = 2 * dpr;
  for (let i = 1; i < d.length; i++) {
    g.strokeStyle = col[d[i][2]] || css("--fg");
    g.beginPath(); g.moveTo(X(d[i - 1][0]), Y(d[i - 1][1])); g.lineTo(X(d[i][0]), Y(d[i][1])); g.stroke();
  }
  g.font = `${11 * dpr}px system-ui`;
  for (const [t, idx, score] of state.repMarks) {
    if (tNow - t > 15) continue;
    const x = X(t);
    g.strokeStyle = g.fillStyle = score >= 80 ? css("--good") : score >= 60 ? css("--warn") : css("--bad");
    g.lineWidth = 1 * dpr;
    g.beginPath(); g.moveTo(x, 16 * dpr); g.lineTo(x, h); g.stroke();
    g.fillText(String(idx), x + 3 * dpr, 13 * dpr);
  }
}

// 3D: level the skeleton with the estimated gravity direction, then orbit it.
function levelRotation(up) {
  if (!up) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const n = Math.hypot(up[0], up[1], up[2]) || 1;
  const a = [up[0] / n, up[1] / n, up[2] / n], b = [0, -1, 0];
  const v = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if (Math.hypot(...v) < 1e-8) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const vx = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]];
  const k = 1 / (1 + c);
  const R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let m = 0; m < 3; m++) s += vx[i][m] * vx[m][j];
    R[i][j] += vx[i][j] + s * k;
  }
  return R;
}
const mul = (R, p) => [R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2], R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2], R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2]];

function draw3d(pose, up, live) {
  const c = $("view3d");
  const [w, h, dpr] = fitCanvas(c);
  const g = c.getContext("2d");
  g.clearRect(0, 0, w, h);
  if (!pose) {
    g.fillStyle = css("--muted"); g.font = `${13 * dpr}px system-ui`; g.textAlign = "center";
    g.fillText("waiting for the athlete...", w / 2, h / 2);
    return;
  }
  const Rl = levelRotation(up);
  const ya = (state.yaw * Math.PI) / 180, pi = (14 * Math.PI) / 180;
  const Ry = [[Math.cos(ya), 0, Math.sin(ya)], [0, 1, 0], [-Math.sin(ya), 0, Math.cos(ya)]];
  const Rx = [[1, 0, 0], [0, Math.cos(pi), -Math.sin(pi)], [0, Math.sin(pi), Math.cos(pi)]];
  const lev = pose.map((p) => mul(Rl, p));
  const pts = lev.map((p) => mul(Rx, mul(Ry, p)));
  const s = 0.62 * Math.min(w, h);
  const ys = pts.map((p) => p[1]);
  const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  const P = (p) => [w / 2 + p[0] * s, h / 2 + (p[1] - cy) * s];
  // floor grid at the feet
  const footY = Math.max(lev[3][1], lev[6][1]);
  g.strokeStyle = css("--line"); g.lineWidth = 1 * dpr;
  for (let i = -3; i <= 3; i++) {
    for (const [a, b] of [[[i / 3 * 0.6, footY, -0.6], [i / 3 * 0.6, footY, 0.6]], [[-0.6, footY, i / 3 * 0.6], [0.6, footY, i / 3 * 0.6]]]) {
      const [x1, y1] = P(mul(Rx, mul(Ry, a))), [x2, y2] = P(mul(Rx, mul(Ry, b)));
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
    }
  }
  const [, bad] = badSets(live);
  const L = css("--left"), R = css("--right"), B = css("--bad"), F = css("--fg");
  const order = H36_BONES.map((b, i) => i).sort((i, j) => (pts[H36_BONES[j][0]][2] + pts[H36_BONES[j][1]][2]) - (pts[H36_BONES[i][0]][2] + pts[H36_BONES[i][1]][2]));
  g.lineWidth = 5 * dpr; g.lineCap = "round";
  for (const i of order) {
    const [a, b] = H36_BONES[i];
    g.strokeStyle = bad.has(a) || bad.has(b) ? B : H36_SIDE[b] > 0 ? L : H36_SIDE[b] < 0 ? R : F;
    const [x1, y1] = P(pts[a]), [x2, y2] = P(pts[b]);
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
  }
}

// drag to rotate the 3D view
(() => {
  const c = $("view3d");
  let x0 = 0;
  c.addEventListener("pointerdown", (e) => { state.dragging = true; x0 = e.clientX; c.setPointerCapture(e.pointerId); });
  c.addEventListener("pointermove", (e) => {
    if (!state.dragging) return;
    state.yaw += (e.clientX - x0) * 0.6; x0 = e.clientX;
    if (state.last) draw3d(state.last.pose3d, state.last.up, state.last.live);
  });
  c.addEventListener("pointerup", () => (state.dragging = false));
})();

// ------------------------------------------------------------------ analyze
$("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = $("file").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  fd.append("exercise", $("exercise2").value);
  $("uploadBtn").disabled = true;
  $("progress").hidden = false;
  $("progressText").textContent = "uploading...";
  $("result").innerHTML = "";
  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    pollJob((await r.json()).id);
  } catch (err) {
    $("progressText").textContent = "failed: " + err.message;
    $("uploadBtn").disabled = false;
  }
});

async function pollJob(id) {
  const j = await (await fetch("/api/jobs/" + id)).json();
  $("bar").style.width = Math.round((j.progress || 0) * 100) + "%";
  $("progressText").textContent = j.state === "failed" ? "failed: " + j.error : `${j.stage} ${Math.round((j.progress || 0) * 100)}%`;
  if (j.state === "done") { $("uploadBtn").disabled = false; showResult(j); return; }
  if (j.state === "failed") { $("uploadBtn").disabled = false; return; }
  setTimeout(() => pollJob(id), 700);
}

function showResult(j) {
  const faults = (j.faults || []).map((f) => `<div class="fault"><b>${esc(f.fault)}</b> - ${f.reps} rep(s)<div class="muted">Cue: ${esc(f.cue)}</div></div>`).join("") || "<p>No form faults detected.</p>";
  $("result").innerHTML = `<div class="result"><video src="${j.video}" controls playsinline></video>
    <div class="card"><div class="card-title">Result</div><div class="score">${j.reps} reps</div>
    <p>Mean form score: <b>${j.score ?? "-"}</b></p>${faults}<p><a href="${j.report}" target="_blank" rel="noopener">Full report</a></p></div></div>`;
}

async function loadSamples() {
  try {
    const samples = await (await fetch("/api/samples")).json();
    if (!samples.length) return;
    $("samplesCard").hidden = false;
    $("samples").innerHTML = samples.filter((s) => s.annotated).map((s) => `<div><video src="/samples/${s.annotated}" controls playsinline preload="metadata"></video>
      <div><b>${esc(s.title)}</b> <span class="muted">${esc(s.note || "")}</span></div></div>`).join("");
  } catch (e) { /* none */ }
}

async function loadSessions() {
  const list = await (await fetch("/api/sessions")).json();
  $("sessions").innerHTML = list.length ? list.map((s) => `<div class="card session">
    ${s.video ? `<video src="${s.video}" controls playsinline preload="metadata"></video>` : ""}
    <h3>${esc(s.exercise || "")} - ${s.reps ?? 0} reps</h3>
    <div class="muted">${esc(s.id)} - ${s.mode} - score ${s.score ?? "-"}</div>
    <div>${s.faults.map(esc).join(", ") || "no faults"}</div>
    ${s.report ? `<a href="${s.report}" target="_blank" rel="noopener">report</a>` : ""}</div>`).join("") : "<p class='muted'>No sessions yet. Finish a live set or analyse a video.</p>";
}

init();
