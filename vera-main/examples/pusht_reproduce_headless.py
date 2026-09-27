"""Headless driver for the PushT reproduction protocol (non-notebook twin of
examples/pusht_dfot_stack.ipynb, cells 2-6) — connects to an already-running
`vera.server.start_vera_server --embodiment pusht` and rolls out a list of
start states x repeats, reporting SR/TP.

Usage:
  python examples/pusht_reproduce_headless.py --port 8820 \
      --frame-indices 3664,11140,6477 --n-repeats 10 --out results.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import zarr

p = argparse.ArgumentParser()
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, required=True)
p.add_argument("--zarr-path", default="/data/scene-rep/u/sizheli/data/pusht_original/pusht_cchi_v7_replay.zarr")
p.add_argument("--frame-indices", required=True, help="comma-separated state indices")
p.add_argument("--n-repeats", type=int, default=10)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--success-threshold", type=float, default=0.9)
p.add_argument("--horizon", type=int, default=200)
p.add_argument("--out", required=True)
p.add_argument("--output-dir", default="outputs/vera_pusht_repro")
p.add_argument("--save-videos", action="store_true")
args = p.parse_args()

from vera.controller.run_mimicgen_eval import RemotePolicy
from vera.datasets.core.pusht_zarr import _patch_zarr_v2_codec_bug
from vera.env_runner.pusht_runner import PushTRunner, PushtRunnerCfg
from vera.server.protocol.websocket_policy_client import WebsocketClientPolicy

_patch_zarr_v2_codec_bug()


def _seed_everything(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


client = WebsocketClientPolicy(host=args.host, port=args.port)
meta = client.get_server_metadata()
view_keys = list(meta["view_keys"])
context_frames = int(meta.get("context_frames", 9))
print(f"planner={meta.get('planner_model')} idm={meta.get('idm_model')} "
      f"views={view_keys} context_frames={context_frames}", flush=True)

runner_cfg = PushtRunnerCfg(
    env_name="pusht", n_repeat=1, num_env_train=1, num_env_eval=0,
    max_episode_steps=args.horizon, action_scale=1.0,
    output_dir=args.output_dir, save_videos=args.save_videos, save_trajectory=False,
    save_rrd=False, video_fps=5,
)
runner = PushTRunner(runner_cfg, device=torch.device("cpu"))
remote = RemotePolicy(client, view_keys=view_keys, view_widths=[252] * len(view_keys),
                       context_frames=context_frames, prompt=None)

group = zarr.open_group(args.zarr_path, mode="r")
states = group["data"]["state"]
indices = [int(x) for x in args.frame_indices.split(",") if x.strip() != ""]
ENV_THRESHOLD = 0.82

out_path = Path(args.out)
done = set()
if out_path.exists():
    for line in out_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["frame_idx"], r["repeat"]))
print(f"resume: {len(done)} episodes already done", flush=True)

with out_path.open("a") as f:
    for fi in indices:
        for rep in range(1, args.n_repeats + 1):
            if (fi, rep) in done:
                continue
            _seed_everything(args.seed)
            reset_to_state = np.asarray(states[fi], dtype=np.float32)
            out = runner.run(remote, options={"reset_to_state": reset_to_state},
                              run_tag=f"state_{fi}_rep{rep}")
            mrm = float(out.get("max_reward_mean", 0.0))
            coverage = min(mrm, 1.0) * ENV_THRESHOLD
            success = int(mrm >= args.success_threshold)
            video_path = None
            if args.save_videos and out.get("save_dir"):
                cand = Path(out["save_dir"]) / "videos" / "policy_env0.mp4"
                video_path = str(cand) if cand.exists() else None
            row = {"frame_idx": fi, "repeat": rep, "max_reward": mrm,
                   "coverage": coverage, "success": success,
                   "save_dir": out.get("save_dir"), "video_path": video_path}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  state {fi:>6d} rep{rep}: max_reward={mrm:.4f} "
                  f"coverage={coverage:.4f} success={success}", flush=True)

rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
sr = 100.0 * np.mean([r["success"] for r in rows])
tp = 100.0 * np.mean([r["max_reward"] for r in rows])
print("=" * 60)
print(f"PushT SR = {sr:.1f}%  (max_reward >= {args.success_threshold}, n={len(rows)})  "
      f"mean max_reward = {tp:.1f}%", flush=True)
