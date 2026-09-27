# Training VERA

**VERA** (*Video-to-Embodied Robot Action model*, from *"Turning Video Models into Generalist Robot
Policies"*) is two stages that **train independently** and share one self-contained dataset core:

- **Video planner** (`vera.video_model` / `vera.idm.dfot`) — an **action-free** diffusion world model that
  dreams the future from the current observation (+ optional text). Embodiment-agnostic; trained once.
- **Jacobian IDM** (`vera.idm` + `vera.policy`) — a faithful, data-efficient translator from the dreamed
  future to actions, built on the robot's Jacobian. Embodiment-specific; trained per robot from self-play.

Decoupling them is the point: the planner never sees actions, and each IDM can be (re)trained without
touching the planner. Both train through one Hydra entry point, `python -m vera.main`, and the data path
is **fully self-contained** (no external sibling-repo imports).

> This release documents training for **MimicGen** and **PushT**; the Allegro / IIWA / DROID configs are
> in-tree but ship no hosted checkpoints or packed datasets yet (they land with **Wave 2**).

## Setup

```bash
pip install -e ".[idm,video,eval]"        # both stages + the simulators
export VERA_DATA_PREFIX=/path/to/data     # root holding your packed datasets
```

`vera.main` composes a run from `vera/configurations/`: a base `config.yaml` + an `experiment`, `dataset`,
`algorithm`, and `algorithm/model`, set via each `config_*.yaml`'s `defaults:` list. Override anything on
the CLI (`experiment.training.batch_size=8 wandb.mode=disabled`). Multi-GPU is automatic (all visible
GPUs, DDP by default; `experiment.strategy=fsdp` for models too big to replicate).

## Stage 1 — Jacobian IDM (per embodiment)

| Environment | `--config-name` | dataset / model |
|---|---|---|
| **MimicGen** (Panda, 2 views) | `config_jacobian_mimicgen_vggt_v3_taskbalanced` | `mimicgen_packed_v3` / `image_jacobian` + VGGT |
| **PushT** (2-DOF, 1 view) | `config_pusht_vggt_fusion_jacobian` | `pusht_packed` / `image_jacobian` + VGGT-fusion |

```bash
python -m vera.main --config-name=config_jacobian_mimicgen_vggt_v3_taskbalanced
python -m vera.main --config-name=config_pusht_vggt_fusion_jacobian
```

The IDM dataset yields views-separate `rgb [T,V,3,H,W]`, `flow`, and the per-embodiment `du` action (see
`vera/datasets/core/actions.py`). The Jacobian IDM regresses the local action-to-flow map and inverts it at
inference — data-efficient and scalable to high-DoF action spaces.

### Get the training data (PushT)

The packed PushT training set (206 episodes, ~1.9 GB — 504×504 JPEG frames + MegaFlow optical flow,
qint8-quantized) ships in the same HF repo; the config expects it at
`$VERA_DATA_PREFIX/datasets/jacobian/pusht_packed`:

```bash
pip install -U huggingface_hub   # 1.x REQUIRED: hub <=0.36 truncates this repo's file
                                 # listing and the --include below matches 0 files
hf download sizhe-lester-li/VERA --include "pusht-packed/*" --local-dir ./vera-ckpts
mkdir -p $VERA_DATA_PREFIX/datasets/jacobian
ln -s "$(pwd)/vera-ckpts/pusht-packed" $VERA_DATA_PREFIX/datasets/jacobian/pusht_packed
```

> If the download prints `Fetching 0 files`, your `huggingface_hub` is too old — upgrade and rerun.

Actions are read at train time from the original PushT replay buffer (not re-hosted — the same
`pusht_cchi_v7_replay.zarr` download as in the [README install section](README.md#install)), so point
`dataset.pusht_zarr_root` at its parent dir:

```bash
python -m vera.main --config-name=config_pusht_vggt_fusion_jacobian \
    dataset.pusht_zarr_root=/path/to/pusht
```

Also available: the large **PushT noise pack** (`pusht-noise-packed/`, 18,685 random-exploration
episodes, ~61 GB) used for IDM pretraining / data-efficiency ablations. Same format, expected at
`$VERA_DATA_PREFIX/datasets/jacobian/pusht_noise_packed`; its actions are **self-contained** (state
deltas from the in-NPZ `traj_state` — no zarr needed), wired up by
`vera/configurations/dataset/pusht_noise_packed.yaml`:

```bash
hf download sizhe-lester-li/VERA --include "pusht-noise-packed/*" --local-dir ./vera-ckpts   # ~61 GB
ln -s "$(pwd)/vera-ckpts/pusht-noise-packed" $VERA_DATA_PREFIX/datasets/jacobian/pusht_noise_packed
```

To regenerate either pack from scratch (or extend it), see `scripts/data/pack_pusht.py` and
[docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) — source-data provenance, exact commands,
MegaFlow install, and GPU requirements for both byte sets.

## Stage 2 — Video planner (WAN / OMNI)

```bash
# single-GPU smoke (T2V 1.3B):
python -m vera.main --config-name=config_wan_combined_4env experiment.training.batch_size=1
# the 14B I2V (OMNI) shards optimizer state across GPUs with FSDP:
python -m vera.main --config-name=config_wan_combined_4env experiment.strategy=fsdp
```

The cross-embodiment **OMNI** planner trains on the **`combined_4env`** mixture — **Allegro-Sim +
Allegro-Real + MimicGen + DROID** — episode-balanced, each kept at native fps/aspect and black-padded to a
common **576-wide** multiview canvas. The video dataset yields the WAN contract
`videos [T,3,128,576] ∈ [-1,1]` + `prompts` (live UMT5 encode). (PushT ships with a small **DFoT** flow
predictor; the **`combined_5env`** mixture — `config_wan_combined_5env.yaml` — adds PushT as a fifth WAN
subset for the upcoming OMNI+PushT checkpoint.)

## Serving (inference)

Once trained (or using the released checkpoints), serve either embodiment over a websocket and drive it
from the example notebooks — see the **quickest-deploy** section of the [README](README.md):

```bash
python -m vera.server.start_vera_server --embodiment pusht    --port 8820 --vis-port 8821
python -m vera.server.start_vera_server --embodiment mimicgen --port 8800 --vis-port 8801 --algo-config <...>
```

## Datasets (self-contained core)

One unified core (`vera/datasets/core/`): `Source` (episode discovery) → `ViewLoader` (packed-JPEG/qint8 or
decord raw-video) → fps-aware `frame_sampler` → `layout` adapter (`separate` for the IDM, `tiled` for the
video model) → `ActionModel` (`du`, IDM only). `MixtureDataset` wraps it for the multi-embodiment WAN set.
One loader feeds **both** stages — the only difference is the view layout.
