"""Flatten the reproduction videos into a clean, shippable layout.

Copies each episode's policy_env0.mp4 from its deeply-nested, timestamped
run directory (examples/outputs/<shard>/run_<ts>_state_<N>_rep<K>/videos/)
into examples/results/videos/state_<N>_rep<K>.mp4, and writes a companion
jsonl with video_path rewritten to the flat, relative location — this is
what actually gets committed (the raw outputs/ run dirs stay gitignored).
"""
import argparse
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results", required=True)
p.add_argument("--videos-out", required=True)
p.add_argument("--jsonl-out", required=True)
args = p.parse_args()

videos_out = Path(args.videos_out)
videos_out.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
out_rows = []
copied = 0
for r in rows:
    src = r.get("video_path")
    dst_name = f"state_{r['frame_idx']}_rep{r['repeat']}.mp4"
    dst = videos_out / dst_name
    if src and Path(src).exists():
        shutil.copy2(src, dst)
        copied += 1
        r = {**r, "video_path": f"videos/{dst_name}"}
    else:
        r = {**r, "video_path": None}
    r.pop("save_dir", None)  # internal path, not meaningful once shipped
    out_rows.append(r)

Path(args.jsonl_out).write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
print(f"copied {copied}/{len(rows)} videos to {videos_out}")
print(f"wrote {args.jsonl_out}")
