# Data generation — the released PushT packs

How the two hosted PushT training sets on
[huggingface.co/sizhe-lester-li/VERA](https://huggingface.co/sizhe-lester-li/VERA)
were generated, and how to regenerate or extend them. Download instructions live in
[TRAINING.md](../TRAINING.md) ("Get the training data (PushT)"); this doc is only
needed if you want to rebuild the bytes yourself.

## What each pack contains

| | `pusht-packed/` | `pusht-noise-packed/` |
|---|---|---|
| Episodes | 206 (human teleop) | 18,685 (uniform-random exploration) |
| Size | ~1.9 GB | ~61 GB |
| RGB | 504×504 JPEG (quality 90), 10 fps, view `00` | same |
| Optical flow | MegaFlow, `qint8_zstd_npz` codec, native 504×504 | same |
| Trajectory in NPZ | `traj_state` (T,2) + global `traj_q_min`/`traj_q_max` | same |
| Actions at train time | **external**: command stream from `pusht_cchi_v7_replay.zarr` via `dataset.pusht_zarr_root` (`action_mode: pusht_pos_cmd`) | **self-contained**: state finite-difference over the in-NPZ `traj_state` (`action_mode: qpos_delta`, see `vera/configurations/dataset/pusht_noise_packed.yaml`) |
| Motion tracks | skipped | skipped |
| Dataset config | `dataset/pusht_packed.yaml` | `dataset/pusht_noise_packed.yaml` |

Both packs use the same on-disk format (`okto_packed` v1): one
`shard_XXXXXX/<episode_id>.npz` per episode holding JPEG frame entries
(`frame_{i:06d}_view_00`), quantized flow entries (`flow_{i:06d}_view_00`),
`traj_*` arrays and a JSON metadata blob, plus a top-level rich `index.json` and
`format_manifest.json`.

**Noise-pack quirk (matches the released bytes and the internal training runs):**
the DINO-WM state tensor is zero-padded to a fixed 246 rows per episode, while the
rendered videos have the episode's true length (~80–125 frames). Each NPZ therefore
stores more `traj_state` rows than decoded frames, and the per-episode metadata's
`num_frames`/rgb key list reflect the padded 246. Frame `k` aligns with state row
`k`; the loader's rejection-sampling retry (`vera/datasets/base.py`,
`_getitem_action`) discards windows that land past the decoded video, so training
only ever sees correctly aligned frames.

## Regenerating

Both packs are produced by **`scripts/data/pack_pusht.py`** (its module docstring
documents every flag). Generation is two steps: render per-episode videos, then
pack (JPEG + MegaFlow flow + trajectory) into NPZs.

### 1. Get the public source data

* **Standard**: PushT replay buffer from the Diffusion Policy benchmark
  (MIT license) — `wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip`
  → contains `pusht_cchi_v7_replay.zarr`.
* **Noise**: the DINO-WM PushT dataset (Zhou et al.,
  [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) — dataset download
  link in its README). Needed files under its root: `train/states.pth`
  `[18685, 246, 5]`, `train/abs_actions.pth`, `train/seq_lengths.pkl`.

### 2. Render the episode videos (CPU, parallelizable)

Replay each episode in `gym-pusht` (`pip install gym-pusht`, the repo's `eval`
extra) with `render_mode="rgb_array"` at **504×504, 10 fps**, writing
`<source-root>/<episode_id>/00_rgb.mp4`:

* standard: reset to the episode's first replay-buffer state and replay its states;
* noise: reset to `states[ep, 0]` and step through `abs_actions[ep, :seq_lengths[ep]-1]`.

Any mp4 writer works (the internal renders used `mediapy`). Episode ids are
zero-padded decimal (`000000` …), one directory per episode.

### 3. Pack

```bash
# standard (206 episodes, single GPU, a few GPU-hours)
python scripts/data/pack_pusht.py \
    --source-root /path/to/pusht_renders \
    --pusht-zarr-root /path/to/pusht \
    --output-root $VERA_DATA_PREFIX/datasets/jacobian/pusht_packed --views 00

# noise (18,685 episodes — shard across GPUs/jobs; ~2-4 s/episode/GPU)
python scripts/data/pack_pusht.py --dataset-type pusht_noise \
    --source-root /path/to/pusht_noise_renders \
    --pusht-noise-source /path/to/dino_wm_pusht_noise \
    --output-root $VERA_DATA_PREFIX/datasets/jacobian/pusht_noise_packed \
    --views 00 --start-index 0 --num-episodes 500 --skip-index
# ... more shards with --start-index 500, 1000, ... then once at the end
# (already-written episodes are skipped; this pass only writes index + manifest):
python scripts/data/pack_pusht.py --dataset-type pusht_noise \
    --source-root /path/to/pusht_noise_renders \
    --pusht-noise-source /path/to/dino_wm_pusht_noise \
    --output-root $VERA_DATA_PREFIX/datasets/jacobian/pusht_noise_packed \
    --views 00 --skip-flow
```

All codec defaults (`--rgb-codec jpeg --rgb-quality 90 --flow-codec qint8_zstd_npz
--megaflow-window-size 4 --megaflow-iters 8`) match the released byte sets; the
internal noise pack was generated with exactly these settings sharded 500
episodes/job. The packer is preemption-safe (atomic NPZ writes, existing episodes
skipped on restart).

### MegaFlow install + GPU requirements

Optical flow needs [MegaFlow](https://github.com/cvg/megaflow) and a CUDA GPU:

```bash
pip install git+https://github.com/cvg/megaflow.git
# on Python 3.11: pip install -e megaflow --ignore-requires-python
```

Weights (`megaflow-flow`) auto-download from HuggingFace on first use. Any ≥16 GB
CUDA GPU works at 504×504 (the released packs were generated on a mix of 24–48 GB
cards; bf16 autocast on Ampere+). Everything except flow runs on CPU — pass
`--skip-flow` to pack RGB + trajectory only, no GPU needed.
