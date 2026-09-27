# PushT Reproduction

## Checkpoints

This result uses the **PushT-specialist pipeline** — a DFoT U-Net3D flow planner (`pusht-dfot`)
paired with a per-frame Jacobian IDM (`pusht-idm`), **not** the OMNI/WAN cross-embodiment
generalist. Both are hosted at [huggingface.co/sizhe-lester-li/VERA](https://huggingface.co/sizhe-lester-li/VERA):

```bash
huggingface-cli download sizhe-lester-li/VERA --include "pusht-dfot/*" --local-dir vera-ckpts
huggingface-cli download sizhe-lester-li/VERA --include "pusht-idm/*"  --local-dir vera-ckpts
```

Dataset: the standard Diffusion-Policy PushT replay buffer
(`pusht_cchi_v7_replay.zarr`, 206 episodes / 25,650 frames, 33 MB) — the source most
commonly linked from the [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
project page / `lerobot`'s pusht dataset mirrors.

## The 16 initial states

```python
FRAME_INDICES = [2590, 2849, 6477, 9585, 10622, 11140, 11658, 12176,
                  16322, 17099, 18653, 19171, 19949, 23058, 25130, 25389]
```

Each is a frame index into the zarr's `data/state` array (an `(x, y, θ)`-style PushT
state), used to reset the environment via `reset_to_state`. 10 trials/state, seed 42.

## Reproduction command

Start the server with the recipe knobs (see [Environment notes](#environment-notes) for
one of them — `TRACKER_BACKEND` — that has no released default matching this recipe):

```bash
export VERA_PUSHT_TRACKER_BACKEND=megaflow   # server default is alltracker
export VERA_PUSHT_N_ACTION_STEPS=3            # server default is 2 (executes fewer of the
                                               # 3 planned steps per replan than this recipe)
python -m vera.server.start_vera_server --embodiment pusht --port 8820 --vis-port 8821 \
    --sample-steps 70                          # server default is 100
```

Then, from `examples/`, either run `pusht_dfot_stack.ipynb` with
`FRAME_INDICES = [2590, 2849, 6477, 9585, 10622, 11140, 11658, 12176, 16322, 17099,
18653, 19171, 19949, 23058, 25130, 25389]` and `N_REPEATS = 10`, or the headless
equivalent:

```bash
python examples/pusht_reproduce_headless.py --port 8820 \
    --frame-indices 2590,2849,6477,9585,10622,11140,11658,12176,16322,17099,18653,19171,19949,23058,25130,25389 \
    --n-repeats 10 --seed 42 --success-threshold 0.9 --horizon 200 \
    --zarr-path /path/to/pusht_cchi_v7_replay.zarr \
    --out results/wide_aligned_repro.jsonl
```

SR = `mean(max_reward >= 0.9)` over all 160 episodes; TP = `mean(max_reward)`.

## Environment notes

Three gaps were found (and fixed in this repo) while getting this recipe to actually run
end to end on the released checkpoints — worth knowing if reproduction still fails:

1. **`gym-pusht` is an install extra, not core.** `pip install -e ".[eval]"` (or at
   minimum `gym-pusht==0.1.5 pymunk<7 gymnasium==0.29.1`) — omitted, the server import
   chain fails with `ModuleNotFoundError: No module named 'gym_pusht'`.
2. **`tracker_backend` was not exposed as a server knob** before this fix
   (`vera/server/start_server_pusht.py`) — the server always used `PlannerCfg`'s default
   (`alltracker`), while this recipe needs `megaflow`. Now controllable via
   `VERA_PUSHT_TRACKER_BACKEND`.
3. **`zarr>=3.0` (the pyproject.toml pin) has no Python-3.10-compatible stable release** —
   every zarr 3.x wheel from `3.0.0` onward requires Python ≥3.11, so `pip` on this
   repo's stated `>=3.10` resolves to a pre-3.0 pre-release (`3.0.0a5` at the time of
   writing). That pre-release has a real bug reading this dataset's v2-format
   Blosc-compressed arrays (`TypeError: 'Blosc' object is not iterable`, reproduces
   regardless of `numcodecs` version) and also misdetects the store as a single array
   instead of a group. Both are worked around in `vera/datasets/core/pusht_zarr.py`
   (`_patch_zarr_v2_codec_bug()` + explicit `zarr.open_group()`) and in the notebook/
   headless driver — no action needed unless you hit this outside those code paths, in
   which case call `_patch_zarr_v2_codec_bug()` before your own `zarr.open_group(...)`.
   The real fix is either a Python ≥3.11 environment or waiting on upstream zarr to ship
   a 3.10-compatible stable release; this patch is a stopgap.
