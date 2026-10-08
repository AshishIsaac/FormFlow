"""Self-contained HTML report for an analysed video (no external assets; light and dark themes)."""

from __future__ import annotations

import html
from pathlib import Path

CSS = """
:root{--bg:#f6f5f2;--card:#fff;--fg:#1d1c1a;--muted:#6b6862;--line:#e3e0da;--good:#2f9e5b;--warn:#c98a00;--bad:#d1453b;
--accent:#2f6fd6;--sig:#2f6fd6}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1d1b;--fg:#ecebe7;--muted:#a19d95;--line:#33312d;
--good:#4cc27a;--warn:#e0a524;--bad:#ef6a5f;--accent:#6b9cf0;--sig:#6b9cf0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 60px}h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card b{display:block;font-size:28px;font-variant-numeric:tabular-nums}.card span{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums;vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}tr:last-child td{border-bottom:0}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;color:#fff}
.good{background:var(--good)}.warn{background:var(--warn)}.bad{background:var(--bad)}
.fault{display:flex;justify-content:space-between;gap:12px;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:10px 14px;margin-bottom:8px}.fault .cue{color:var(--muted)}
video{width:100%;border-radius:12px;background:#000;margin-top:8px}svg{width:100%;height:auto;display:block}
.chart{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px}
"""


def _cls(score: float | None) -> str:
    s = score or 0
    return "good" if s >= 80 else "warn" if s >= 60 else "bad"


def _chart(values: list, reps: list[dict], fps: float, w: int = 1000, h: int = 220) -> str:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return "<p class='muted'>No signal.</p>"
    lo, hi = min(vals), max(vals)
    hi = hi if hi - lo > 1e-6 else lo + 1
    n = len(values)
    pad = 28

    def X(i):
        return pad + (w - 2 * pad) * i / max(n - 1, 1)

    def Y(v):
        return h - pad - (h - 2 * pad) * (v - lo) / (hi - lo)

    parts = [f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='rep signal over time'>"]
    for r in reps:
        x0, x1 = X(r["start_frame"]), X(r["end_frame"])
        col = {"good": "var(--good)", "warn": "var(--warn)", "bad": "var(--bad)"}[_cls(r.get("score"))]
        parts.append(f"<rect x='{x0:.1f}' y='{pad - 8}' width='{max(x1 - x0, 1):.1f}' height='{h - 2 * pad + 8}' "
                     f"fill='{col}' opacity='0.13' rx='3'/>")
        parts.append(f"<text x='{(x0 + x1) / 2:.1f}' y='{pad - 12}' font-size='12' text-anchor='middle' "
                     f"fill='var(--muted)'>{r['index']}</text>")
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(values) if v is not None)
    parts.append(f"<polyline points='{pts}' fill='none' stroke='var(--sig)' stroke-width='2' stroke-linejoin='round'/>")
    for k in range(0, int(n / fps) + 1, max(1, int(n / fps / 10))):
        x = X(k * fps)
        parts.append(f"<text x='{x:.1f}' y='{h - 8}' font-size='11' text-anchor='middle' fill='var(--muted)'>{k}s</text>")
    parts.append(f"<text x='{pad}' y='{h - pad + 14}' font-size='11' fill='var(--muted)'>{lo:.0f}</text>")
    parts.append(f"<text x='{pad}' y='{pad + 4}' font-size='11' fill='var(--muted)'>{hi:.0f}</text>")
    parts.append("</svg>")
    return "".join(parts)


def write_html(result: dict, path: Path) -> None:
    s = result["summary"]
    e = html.escape
    reps = s.get("rep_list", [])
    rows = []
    for r in reps:
        faults = "<br>".join(f"{e(f['fault'])}{' (' + f['side'] + ')' if f.get('side') else ''} "
                             f"<span class='muted'>{f['value']:.2f} vs {f['limit']:.2f}</span>" for f in r["faults"]) \
            or "<span class='muted'>clean</span>"
        rows.append(f"<tr><td>{r['index']}</td><td><span class='pill {_cls(r['score'])}'>{r['score']:.0f}</span></td>"
                    f"<td>{r['out_s']:.2f}s / {r['back_s']:.2f}s</td><td>{r['peak_value']:.0f}</td><td>{faults}</td></tr>")
    faults = "".join(f"<div class='fault'><div><b>{e(f['fault'])}</b><div class='cue'>Cue: {e(f['cue'])}</div></div>"
                     f"<div class='muted'>{f['reps']} of {s['reps']} reps</div></div>" for f in s.get("faults", [])) \
        or "<p class='muted'>No form faults detected.</p>"
    video = f"<h2>Annotated video</h2><video src='{e(result['annotated'])}' controls playsinline></video>" \
        if result.get("annotated") else ""
    score = s.get("mean_score")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FormFlow report</title><style>{CSS}</style></head>
<body><main>
<h1>{e(result['display'])} - FormFlow report</h1>
<div class="muted">{e(Path(result['video']).name)} - {result['frames']} frames at {result['fps']:.0f} fps - {e(result['mode'])} analysis
- rep signal: {e(s['signal'].replace('_', ' '))}</div>
<div class="cards">
<div class="card"><b>{s['reps']}</b><span>reps counted</span></div>
<div class="card"><b>{'-' if score is None else f'{score:.0f}'}</b><span>mean form score / 100</span></div>
<div class="card"><b>{s['partial_reps']}</b><span>partial reps (not counted)</span></div>
<div class="card"><b>{'-' if s.get('mean_rep_s') is None else f"{s['mean_rep_s']:.1f}s"}</b><span>mean rep time</span></div>
</div>
<h2>What to work on</h2>{faults}
<h2>Rep signal</h2><div class="chart">{_chart(result['signal']['values'], reps, result['fps'])}</div>
{video}
<h2>Every rep</h2>
<table><thead><tr><th>#</th><th>score</th><th>out / back</th><th>peak</th><th>faults</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan=5 class=muted>No reps counted.</td></tr>'}</tbody></table>
<p class="muted" style="margin-top:24px">Models: {e(', '.join(f'{k}={v}' for k, v in result.get('models', {}).items()))}.
Joint angles come from a monocular 3D estimate (learned body prior), not a motion-capture lab: treat small differences
(a few degrees) as noise.</p>
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")
