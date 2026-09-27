"""Build a static HTML viewer for the PushT reproduction videos.

Reads a results jsonl (rows with frame_idx, repeat, max_reward, success,
video_path) and emits a self-contained-except-videos HTML page: one section
per state (grouped by the documented hard/easy/undocumented bucket), one
video tile per repeat, pass/fail colored. Videos are referenced by relative
path (NOT base64-embedded — at ~160 videos that would make an unusable
multi-hundred-MB single file) so serve this page from the same root the
video paths are relative to, e.g.:

  python examples/build_pusht_viewer.py \
      --results examples/results/wide_aligned_repro_video.jsonl \
      --root examples \
      --out examples/results/viewer.html
  cd examples && python -m http.server 8000
  # open http://localhost:8000/results/viewer.html
"""
import argparse
import html
import json
from pathlib import Path

HARD = {17099, 19171}
EASY = {2590, 2849, 6477, 10622, 11140, 11658, 16322, 18653, 23058, 25130}
# everything else falls into "undocumented"

p = argparse.ArgumentParser()
p.add_argument("--results", required=True, help="jsonl with video_path per episode")
p.add_argument("--root", required=True, help="dir the output HTML will be served relative to")
p.add_argument("--out", required=True)
p.add_argument("--title", default="PushT reproduction — episode viewer")
args = p.parse_args()

rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
root = Path(args.root).resolve()
out_path = Path(args.out)

by_state = {}
for r in rows:
    by_state.setdefault(r["frame_idx"], []).append(r)
for state in by_state:
    by_state[state].sort(key=lambda r: r["repeat"])


def bucket_of(state):
    if state in HARD:
        return "hard", "labeled hard (“previously sub-90%”)"
    if state in EASY:
        return "easy", "labeled easy (“previously ≥90%”)"
    return "undoc", "undocumented rationale"


def rel_video(row):
    vp = row.get("video_path")
    if not vp:
        return None
    if not Path(vp).is_absolute():
        # already relative-to-the-jsonl's-directory by construction (e.g. the
        # flattened "videos/state_N_repK.mp4" shipped layout) — use as-is,
        # since resolving it against the CWD would silently point nowhere.
        return vp
    try:
        return str(Path(vp).resolve().relative_to(out_path.resolve().parent))
    except ValueError:
        # fall back to relative-to-root if the viewer isn't nested under root's parent
        try:
            return "../" + str(Path(vp).resolve().relative_to(root))
        except ValueError:
            return None


n = len(rows)
succ = sum(r["success"] for r in rows)
sr = 100.0 * succ / n if n else 0.0
tp = 100.0 * sum(r["max_reward"] for r in rows) / n if n else 0.0

bucket_order = ["hard", "easy", "undoc"]
bucket_label = {k: v for k, v in (bucket_of(s) for s in by_state)}
states_by_bucket = {k: [] for k in bucket_order}
for state in sorted(by_state):
    b, _ = bucket_of(state)
    states_by_bucket[b].append(state)

CSS = """
:root {
  --bg: #0b0d12; --panel: #12151c; --panel2: #171b24; --line: #232838;
  --text: #e7e9ee; --muted: #8b93a7; --accent: #6ea8fe;
  --ok: #34d399; --ok-bg: #0f2a20; --fail: #f87171; --fail-bg: #2a1214;
}
@media (prefers-color-scheme: light) {
  :root { --bg:#f6f7fb; --panel:#fff; --panel2:#f0f2f7; --line:#e3e6ee;
    --text:#171a21; --muted:#5c6470; --accent:#2f6fed;
    --ok:#0a8f5b; --ok-bg:#e6f7ef; --fail:#c22; --fail-bg:#fdeceb; }
}
:root[data-theme="dark"] { color-scheme: dark; }
:root[data-theme="light"] { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 -apple-system, "Inter", "Segoe UI", sans-serif;
}
header {
  padding: 28px 32px 20px; border-bottom: 1px solid var(--line);
  position: sticky; top: 0; background: var(--bg); z-index: 5;
}
h1 { font-size: 1.35rem; margin: 0 0 6px; font-weight: 650; letter-spacing: -0.01em; }
.summary { color: var(--muted); font-size: 0.92rem; }
.summary b { color: var(--text); font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 18px; margin-top: 12px; flex-wrap: wrap; }
.legend span { display: inline-flex; align-items: center; gap: 6px; font-size: 0.82rem; color: var(--muted); }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
main { padding: 8px 32px 60px; max-width: 1400px; margin: 0 auto; }
.bucket-title {
  margin: 34px 0 4px; font-size: 1.02rem; font-weight: 650;
}
.bucket-sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 14px; }
.state-block { margin-bottom: 26px; }
.state-head {
  display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px;
}
.state-head .sid { font-variant-numeric: tabular-nums; font-weight: 650; font-size: 0.95rem; }
.state-head .stat { color: var(--muted); font-size: 0.82rem; font-variant-numeric: tabular-nums; }
.grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
}
.tile {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  overflow: hidden;
}
.tile video, .tile .novid {
  width: 100%; aspect-ratio: 1 / 1; display: block; background: #000; object-fit: contain;
}
.tile .novid {
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 0.78rem; background: var(--panel2);
}
.tile .meta {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 9px; font-size: 0.78rem; font-variant-numeric: tabular-nums;
}
.tile.ok .meta { background: var(--ok-bg); color: var(--ok); }
.tile.fail .meta { background: var(--fail-bg); color: var(--fail); }
.tile .rep { color: var(--muted); }
"""

parts = [
    "<!DOCTYPE html>\n"
    '<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    f"<title>{html.escape(args.title)}</title><style>{CSS}</style></head><body>"
]
parts.append("<header><h1>PushT reproduction — episode viewer</h1>")
parts.append(
    f'<div class="summary">n=<b>{n}</b> episodes &nbsp;·&nbsp; '
    f'SR=<b>{sr:.1f}%</b> ({succ}/{n}) &nbsp;·&nbsp; '
    f'mean max_reward=<b>{tp:.1f}%</b> &nbsp;·&nbsp; '
    f"max_reward ≥ 0.9 = success</div>"
)
parts.append(
    '<div class="legend">'
    '<span><i class="dot" style="background:var(--ok)"></i>success</span>'
    '<span><i class="dot" style="background:var(--fail)"></i>failure</span>'
    "</div></header><main>"
)

BUCKET_TITLES = {
    "hard": ("Labeled hard states", "states pre-selected because they scored &lt;90% in an earlier reference run"),
    "easy": ("Labeled easy states", "states pre-selected because they scored ≥90% in an earlier reference run"),
    "undoc": ("Undocumented-rationale states", "no recorded selection criterion — turned out to be the highest-variance bucket"),
}

for b in bucket_order:
    states = states_by_bucket[b]
    if not states:
        continue
    title, sub = BUCKET_TITLES[b]
    b_rows = [r for s in states for r in by_state[s]]
    b_sr = 100.0 * sum(r["success"] for r in b_rows) / len(b_rows)
    parts.append(f'<div class="bucket-title">{title} &nbsp;<span style="color:var(--muted);font-weight:500">'
                  f"({len(states)} states, {len(b_rows)} eps, SR {b_sr:.1f}%)</span></div>")
    parts.append(f'<div class="bucket-sub">{sub}</div>')
    for state in states:
        eps = by_state[state]
        s = sum(r["success"] for r in eps)
        parts.append('<div class="state-block">')
        parts.append(
            f'<div class="state-head"><span class="sid">state {state}</span>'
            f'<span class="stat">{s}/{len(eps)} success</span></div>'
        )
        parts.append('<div class="grid">')
        for r in eps:
            cls = "ok" if r["success"] else "fail"
            vrel = rel_video(r)
            if vrel:
                # preload="metadata" (not "none"): decodes just the first frame so idle
                # tiles show real content instead of a black box, without eagerly
                # downloading all 160 videos on page load.
                media = f'<video src="{html.escape(vrel)}" muted loop playsinline preload="metadata" onmouseover="this.play()" onmouseout="this.pause()"></video>'
            else:
                media = '<div class="novid">no video</div>'
            parts.append(
                f'<div class="tile {cls}">{media}'
                f'<div class="meta"><span class="rep">rep {r["repeat"]}</span>'
                f'<span>{r["max_reward"]:.3f}</span></div></div>'
            )
        parts.append("</div></div>")

parts.append("</main></body></html>")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("".join(parts), encoding="utf-8")
print(f"wrote {out_path} ({len(rows)} episodes, {sum(1 for r in rows if rel_video(r))} with video)")
