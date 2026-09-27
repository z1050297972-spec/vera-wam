"""Pack the PushT datasets into VERA's packed-NPZ training format.

This is the reference generator for TWO hosted dataset folders on
`huggingface.co/sizhe-lester-li/VERA <https://huggingface.co/sizhe-lester-li/VERA>`_,
selected with ``--dataset-type``:

* ``pusht`` (default) -> ``pusht-packed/`` (206 human-teleop episodes, ~1.9 GB) —
  the dataset `config_pusht_vggt_fusion_jacobian` trains on.
* ``pusht_noise``     -> ``pusht-noise-packed/`` (18,685 random-exploration
  episodes, ~61 GB) — the large IDM pretraining / data-efficiency-ablation set.

**Downloading the hosted folders is the recommended path** (see TRAINING.md, "Get
the training data (PushT)"); run this script only to regenerate a pack from
scratch or to extend it (more episodes, other codecs / resolutions). A fuller
provenance walkthrough for both packs lives in ``docs/DATA_GENERATION.md``.

Source data for ``--dataset-type pusht`` (public, Diffusion Policy benchmark):

* **Trajectory + actions** — the original PushT replay buffer
  ``pusht_cchi_v7_replay.zarr`` (`real-stanford/diffusion_policy
  <https://github.com/real-stanford/diffusion_policy>`_;
  ``wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip``).
  This script embeds the per-episode agent state (``traj_state``) and the global
  state bounds (``traj_q_min`` / ``traj_q_max``); the *action* stream stays in the
  zarr and is read at train time via ``dataset.pusht_zarr_root`` — it is NOT
  copied into the NPZs.
* **RGB videos** — per-episode ``gym-pusht`` re-renders of the replay-buffer
  episodes (504x504 @ 10 fps, one ``00_rgb.mp4`` per episode), laid out as
  ``<source-root>/<episode_id>/00_rgb.mp4``. Replay each episode's states in
  ``gym_pusht`` (the `eval` extra) with ``render_mode="rgb_array"`` at 504x504 to
  produce them.

Source data for ``--dataset-type pusht_noise`` (public, DINO-WM):

* **Trajectory** — ``train/states.pth`` from the DINO-WM PushT dataset
  (`gaoyuezhou/dino_wm <https://github.com/gaoyuezhou/dino_wm>`_, see its README
  for the dataset download; tensor ``[N, T_max, 5]``, per-episode true lengths in
  ``seq_lengths.pkl``). Pass its root (the dir whose ``train/`` contains
  ``states.pth``) as ``--pusht-noise-source``. Channels 0-1 (agent x/y) become
  ``traj_state`` (padded to ``T_max`` = 246 rows; the decoded video is shorter —
  frame ``k`` aligns with state row ``k``). Train-time actions are the min-max
  normalized state finite-difference, computed FROM these embedded arrays — no
  external action file is needed (``dataset/pusht_noise_packed.yaml``).
* **RGB videos** — per-episode ``gym-pusht`` replays of the episode's absolute
  actions (``abs_actions.pth``) at 504x504 @ 10 fps, same
  ``<source-root>/<episode_id>/00_rgb.mp4`` layout as above.

Each episode becomes one ``shard_XXXXXX/<episode_id>.npz`` holding JPEG-encoded
frames (``frame_{i:06d}_view_00``), MegaFlow optical flow quantized with the
``qint8_zstd_npz`` codec (``flow_{i:06d}_view_00``), the trajectory arrays
(``traj_*``), and a JSON metadata blob (``__packed_episode_metadata__``). After
packing, the script writes ``format_manifest.json`` and a rich ``index.json``
(shard-relative NPZ paths + inlined per-episode metadata) — everything
``vera.datasets.core`` (PackedSource / PackedViewLoader) needs. The internal
``metadata_cache.json`` shipped alongside earlier packs is a legacy cache the
public loader ignores; it is not emitted here.

Optical flow requires **MegaFlow** and a **CUDA GPU** (a full 206-episode pack is
a few GPU-hours; everything else runs on CPU). MegaFlow is imported lazily — the
script imports and runs ``--help`` without it — and is the same backend the PushT
serving recipe selects via ``VERA_PUSHT_TRACKER_BACKEND=megaflow``
(docs/PUSHT_REPRODUCTION.md). Install it from
`github.com/cvg/megaflow <https://github.com/cvg/megaflow>`_::

    pip install git+https://github.com/cvg/megaflow.git
    # or: git clone https://github.com/cvg/megaflow && \
    #     pip install -e megaflow --ignore-requires-python   # package works on 3.11

Weights (``megaflow-flow``) auto-download from HuggingFace on first use.

Examples::

    # standard pack (206 episodes)
    python scripts/data/pack_pusht.py \
        --source-root /path/to/pusht_renders \
        --pusht-zarr-root /path/to/pusht \
        --output-root $VERA_DATA_PREFIX/datasets/jacobian/pusht_packed \
        --views 00

    # noise pack (18,685 episodes; shard across jobs with --start-index /
    # --num-episodes + --skip-index, then run once more to write the index)
    python scripts/data/pack_pusht.py --dataset-type pusht_noise \
        --source-root /path/to/pusht_noise_renders \
        --pusht-noise-source /path/to/dino_wm_pusht_noise \
        --output-root $VERA_DATA_PREFIX/datasets/jacobian/pusht_noise_packed \
        --views 00

Preemption-safe: atomic NPZ writes (temp file + rename), already-written
episodes are skipped on restart, and SIGTERM finishes the current episode before
exiting. The packing logic is transplanted unchanged from the internal
multi-embodiment generator (only the PushT path is kept); the writer + codecs
live in ``vera.utils.droid_packed_format``.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

try:  # optional fast video decode; cv2 fallback below
    import torchvision
except ImportError:  # pragma: no cover
    torchvision = None

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover - dependency-free environments
    import re

    def _natural_key(value: object) -> list[object]:
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))
        ]

    def natsorted(iterable, *, key=None):
        if key is None:
            return sorted(iterable, key=_natural_key)
        return sorted(iterable, key=lambda item: _natural_key(key(item)))

# Packed-NPZ writer + codecs — shipped with vera (vera/utils/droid_packed_format.py).
from vera.utils.droid_packed_format import (
    DEFAULT_RGB_CODEC,
    DEFAULT_RGB_QUALITY,
    PackedEpisodeMetadata,
    encode_flow_payload,
    encode_image_bytes,
    flow_entry_key,
    metadata_to_uint8_array,
    rgb_entry_key,
    shard_relative_npz_path,
    trajectory_entry_key,
    write_npz_stored_atomic,
)

PACKED_METADATA_KEY = "__packed_episode_metadata__"
FLOW_BACKEND = "megaflow"

# ---------------------------------------------------------------------------
# Graceful SIGTERM shutdown (SLURM preemption etc.)
# ---------------------------------------------------------------------------
_SHUTDOWN_REQUESTED = False


def _sigterm_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print("[signal] SIGTERM received — finishing current episode then exiting", flush=True)


signal.signal(signal.SIGTERM, _sigterm_handler)


# ============================================================================
# PushT adapter — episode discovery + trajectory loading from the replay zarr
# ============================================================================


class PushTAdapter:
    """Discovers rendered episode dirs and slices the global replay zarr.

    The zarr concatenates all episodes (``data/state``, split by
    ``meta/episode_ends``); each episode's state slice plus the GLOBAL per-channel
    state bounds (``q_min``/``q_max`` over the whole buffer) are embedded in the
    NPZ. Actions are intentionally NOT embedded (train-time reads them from the
    zarr via ``dataset.pusht_zarr_root``).
    """

    def __init__(self, args: argparse.Namespace):
        import zarr  # lazy: only the packer needs it

        # Same zarr-v2-store workaround the train-time reader uses
        # (vera/datasets/core/pusht_zarr.py): plain zarr.open() on the pre-3.0
        # zarr pre-releases misdetects this v2 directory store, and the V2
        # Blosc-decode path needs the numcodecs get_codec patch.
        from vera.datasets.core.pusht_zarr import _patch_zarr_v2_codec_bug

        _patch_zarr_v2_codec_bug()
        self.args = args
        zarr_root = Path(args.pusht_zarr_root)
        if zarr_root.suffix != ".zarr":
            zarr_root = zarr_root / "pusht_cchi_v7_replay.zarr"
        self._zarr = zarr.open_group(str(zarr_root), mode="r")
        joint_indices = [0, 1]
        # Materialize once (like PushtZarr): PushT is 2-DOF / ~25k rows, trivial.
        self._state = np.asarray(self._zarr["data"]["state"])
        state = self._state[:, joint_indices]
        self._q_min = state.min(axis=0).astype(np.float32)
        self._q_max = state.max(axis=0).astype(np.float32)
        self._episode_ends = np.asarray(self._zarr["meta"]["episode_ends"])
        self._joint_indices = joint_indices

    def discover_episodes(self, source_root: Path) -> list[dict[str, str]]:
        episode_dirs = natsorted(
            [d for d in source_root.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        return [
            {"relative_path": d.name, "episode_id": d.name}
            for d in episode_dirs
        ]

    def get_videos(self, ep_dir: Path, views: list[str]) -> dict[str, Path]:
        videos: dict[str, Path] = {}
        for view in views:
            mp4 = ep_dir / f"{view}_rgb.mp4"
            if mp4.exists():
                videos[view] = mp4
        return videos

    def load_trajectory(self, ep_dir: Path, episode_id: str) -> dict[str, np.ndarray]:
        ep_idx = int(episode_id)
        start = 0 if ep_idx == 0 else int(self._episode_ends[ep_idx - 1])
        end = int(self._episode_ends[ep_idx])
        ji = np.asarray(self._joint_indices, dtype=np.int64)
        state_slice = np.asarray(
            self._state[start:end][:, ji],
            dtype=np.float32,
        )
        return {
            "state": state_slice,
            "q_min": self._q_min,
            "q_max": self._q_max,
        }

    def get_num_frames_from_trajectory(self, trajectory_arrays: dict[str, np.ndarray]) -> int:
        return int(trajectory_arrays["state"].shape[0])


# ============================================================================
# PushT noise adapter — trajectory from the DINO-WM states.pth
# ============================================================================


class PushTNoiseAdapter:
    """Discovers rendered episode dirs and slices the DINO-WM ``states.pth``.

    ``states.pth`` is a single ``[N, T_max, 5]`` tensor (all N=18,685 episodes,
    zero-padded to T_max=246 rows); channels 0-1 are the agent x/y state. Each
    episode embeds its full (padded) ``[T_max, 2]`` slice as ``traj_state`` plus
    the GLOBAL per-channel bounds over the whole tensor (``traj_q_min`` /
    ``traj_q_max``) — exactly what the train-time state-delta action model
    (``dataset/pusht_noise_packed.yaml``) consumes. Nothing else is needed:
    unlike the standard PushT pack there is no external action stream.

    Faithful transplant of the internal ``pusht_noise`` packer adapter —
    including the torch advanced-indexing form ``state[ep, :, [0, 1]]``, which
    in torch keeps the ``[T, 2]`` layout of the released bytes.
    """

    def __init__(self, args: argparse.Namespace):
        noise_source = Path(args.pusht_noise_source)
        state_file = noise_source / "train" / "states.pth"
        self._state = torch.load(state_file, map_location="cpu")
        joint_indices = [0, 1]
        state_ji = self._state[..., joint_indices]
        self._q_min = state_ji.amin(dim=(0, 1)).float().numpy()
        self._q_max = state_ji.amax(dim=(0, 1)).float().numpy()
        self._joint_indices = joint_indices

    def discover_episodes(self, source_root: Path) -> list[dict[str, str]]:
        episode_dirs = natsorted(
            [d for d in source_root.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        return [
            {"relative_path": d.name, "episode_id": d.name}
            for d in episode_dirs
        ]

    def get_videos(self, ep_dir: Path, views: list[str]) -> dict[str, Path]:
        videos: dict[str, Path] = {}
        for view in views:
            mp4 = ep_dir / f"{view}_rgb.mp4"
            if mp4.exists():
                videos[view] = mp4
        return videos

    def load_trajectory(self, ep_dir: Path, episode_id: str) -> dict[str, np.ndarray]:
        ep_idx = int(episode_id)
        ji = self._joint_indices
        state_slice = self._state[ep_idx, :, ji].float().numpy()
        return {
            "state": state_slice,
            "q_min": self._q_min,
            "q_max": self._q_max,
        }

    def get_num_frames_from_trajectory(self, trajectory_arrays: dict[str, np.ndarray]) -> int:
        return int(trajectory_arrays["state"].shape[0])


# ============================================================================
# Video / flow helpers
# ============================================================================


def _load_video_uint8(video_path: Path) -> tuple[np.ndarray, float]:
    if torchvision is not None and hasattr(torchvision.io, "read_video"):
        try:
            video, _, info = torchvision.io.read_video(str(video_path), pts_unit="sec")
            if video.ndim != 4:
                raise RuntimeError(
                    f"Expected 4D video tensor from {video_path}, got {tuple(video.shape)}"
                )
            fps = float(info.get("video_fps", 30.0))
            return video.cpu().numpy(), fps
        except Exception:
            pass

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or np.isnan(fps):
        fps = 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8), fps
    return np.stack(frames, axis=0).astype(np.uint8), fps


def _load_megaflow_model(device: torch.device):
    """Lazily import + load MegaFlow (weights auto-download from HuggingFace)."""
    try:
        from megaflow.model import MegaFlow
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Optical-flow packing requires the `megaflow` package "
            "(github.com/cvg/megaflow):\n"
            "    pip install git+https://github.com/cvg/megaflow.git\n"
            "(pass --ignore-requires-python on Python 3.11). "
            "Alternatively run with --skip-flow to pack RGB + trajectory only."
        ) from exc

    preload_device = "cuda" if device.type == "cuda" else "cpu"
    model = MegaFlow.from_pretrained("megaflow-flow", device=preload_device)
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)
    return model


def _maybe_empty_cache(device: torch.device, *, cache_policy: str, stage: str) -> None:
    if device.type != "cuda":
        return
    if cache_policy == "always":
        torch.cuda.empty_cache()
    elif cache_policy == "episode" and stage == "episode":
        torch.cuda.empty_cache()


def _resize_video_uint8(video_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a THWC uint8 video to (target_h, target_w). No-op if already that size."""
    if video_np.ndim != 4 or video_np.shape[-1] != 3:
        raise ValueError(f"Expected THWC uint8 video, got shape={video_np.shape}")
    if video_np.shape[0] == 0:
        return video_np
    if video_np.shape[1] == target_h and video_np.shape[2] == target_w:
        return video_np
    video_t = (
        torch.from_numpy(video_np)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
    )
    resized = F.interpolate(
        video_t,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.round().clamp_(0, 255).to(dtype=torch.uint8)
    return resized.permute(0, 2, 3, 1).cpu().numpy()


def _compute_megaflow_flows(
    video_np: np.ndarray,
    *,
    megaflow_model,
    device: torch.device,
    megaflow_window_size: int,
    megaflow_iters: int,
    cache_policy: str,
) -> torch.Tensor:
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
    _, num_frames, _, height, width = video.shape
    if num_frames < 2:
        return torch.zeros((0, 2, height, width), dtype=torch.float32, device=device)

    window_size = max(2, int(megaflow_window_size))
    stride = max(1, window_size - 1)
    compute_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    flow_chunks: list[torch.Tensor] = []
    for start in range(0, num_frames - 1, stride):
        end = min(start + window_size, num_frames)
        if end - start < 2:
            continue
        chunk = video[:, start:end]
        with torch.autocast(
            device_type=device.type,
            dtype=compute_dtype,
            enabled=(device.type == "cuda"),
        ):
            results_dict = megaflow_model(chunk, num_reg_refine=int(megaflow_iters))
        flow_chunk = results_dict["flow_preds"][-1][0].detach().float()
        flow_chunks.append(flow_chunk)
        del chunk, results_dict, flow_chunk
        _maybe_empty_cache(device, cache_policy=cache_policy, stage="chunk")

    if not flow_chunks:
        return torch.zeros((0, 2, height, width), dtype=torch.float32, device=device)
    return torch.cat(flow_chunks, dim=0)


def _compute_video_flows(
    video_np: np.ndarray,
    *,
    flow_resolution: tuple[int, int] | None,
    device: torch.device,
    megaflow_model,
    megaflow_window_size: int,
    megaflow_iters: int,
    cache_policy: str,
) -> torch.Tensor:
    """Compute optical flow with MegaFlow. If flow_resolution is set, resize the
    video to that resolution first (flow is stored at this resolution)."""
    if flow_resolution is not None:
        flow_video_np = _resize_video_uint8(video_np, flow_resolution[0], flow_resolution[1])
    else:
        flow_video_np = video_np
    return _compute_megaflow_flows(
        flow_video_np,
        megaflow_model=megaflow_model,
        device=device,
        megaflow_window_size=megaflow_window_size,
        megaflow_iters=megaflow_iters,
        cache_policy=cache_policy,
    )


# ============================================================================
# Episode assembly (entry stream for one NPZ)
# ============================================================================


def _build_base_metadata(
    *,
    episode_id: str,
    source_relative_path: str,
    num_frames: int,
    views: list[str],
    trajectory_arrays: dict[str, np.ndarray],
    rgb_codec: str,
    rgb_quality: int,
) -> PackedEpisodeMetadata:
    metadata = PackedEpisodeMetadata(
        episode_id=episode_id,
        source_relative_path=source_relative_path,
        num_frames=num_frames,
        views=views,
    )
    for view in views:
        metadata.rgb_entries[view] = {
            "codec": rgb_codec,
            "quality": rgb_quality,
            "num_frames": num_frames,
            "keys": [rgb_entry_key(t, view) for t in range(num_frames)],
        }
    for dataset_key, arr in trajectory_arrays.items():
        metadata.trajectory_entries[dataset_key] = {
            "key": trajectory_entry_key(dataset_key),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    return metadata


def _episode_entries(
    *,
    videos: dict[str, Path],
    trajectory_arrays: dict[str, np.ndarray],
    metadata: PackedEpisodeMetadata,
    device: torch.device,
    megaflow_model,
    megaflow_window_size: int,
    megaflow_iters: int,
    skip_flow: bool,
    flow_codec: str,
    flow_resolution: tuple[int, int] | None,
    rgb_codec: str,
    rgb_quality: int,
    cache_policy: str,
    temporal_stride: int = 1,
) -> Iterable[tuple[str, np.ndarray]]:
    for dataset_key, arr in trajectory_arrays.items():
        yield trajectory_entry_key(dataset_key), np.asarray(arr)

    for view, video_path in videos.items():
        video_np, _fps = _load_video_uint8(video_path)
        # Temporal subsampling (e.g. 50Hz -> 10Hz with stride=5); unused (=1) for
        # the released PushT pack, whose renders are already 10 fps.
        if temporal_stride > 1:
            video_np = video_np[::temporal_stride]
        if video_np.shape[0] == 0:
            continue

        for frame_idx in range(video_np.shape[0]):
            payload = encode_image_bytes(
                video_np[frame_idx],
                codec=rgb_codec,
                quality=rgb_quality,
            )
            yield rgb_entry_key(frame_idx, view), np.frombuffer(payload, dtype=np.uint8)

        if not skip_flow:
            flows = _compute_video_flows(
                video_np,
                flow_resolution=flow_resolution,
                device=device,
                megaflow_model=megaflow_model,
                megaflow_window_size=megaflow_window_size,
                megaflow_iters=megaflow_iters,
                cache_policy=cache_policy,
            )
            flows_np = (flows.cpu() if flows.is_cuda else flows).numpy()
            if flow_codec in {"raw_npy", "zlib_npy", "zstd_npy"}:
                flows_np = flows_np.astype(np.float16)
            storage_shape = list(flows_np.shape[1:])
            flow_h = flow_resolution[0] if flow_resolution else video_np.shape[1]
            flow_w = flow_resolution[1] if flow_resolution else video_np.shape[2]
            metadata.flow_entries[view] = {
                "backend": FLOW_BACKEND,
                "codec": flow_codec,
                "dtype": "float32" if flow_codec == "qint8_zstd_npz" else str(flows_np.dtype),
                "num_frames": int(flows_np.shape[0]),
                "frame_shape": storage_shape,
                "storage_frame_shape": storage_shape,
                "source_frame_shape": [2, int(video_np.shape[1]), int(video_np.shape[2])],
                "flow_resolution": [int(flow_h), int(flow_w)],
                "keys": [flow_entry_key(t, view) for t in range(int(flows_np.shape[0]))],
            }
            for frame_idx in range(flows_np.shape[0]):
                payload = encode_flow_payload(flows_np[frame_idx], codec=flow_codec)
                yield flow_entry_key(frame_idx, view), np.frombuffer(payload, dtype=np.uint8)
            del flows_np
            _maybe_empty_cache(device, cache_policy=cache_policy, stage="chunk")
        else:
            metadata.notes.append(f"flow_skipped:{view}")

        # Motion tracks are not part of the PushT pack (the released dataset was
        # generated with them skipped; the training config sets
        # load_motion_tracks: false). Note kept for parity with the released bytes.
        metadata.notes.append(f"motion_tracks_skipped:{view}")

        del video_np

    yield PACKED_METADATA_KEY, metadata_to_uint8_array(metadata)


# ============================================================================
# Index / manifest emission
# ============================================================================


def _write_json_atomic(path: Path, payload, *, indent: int | None = None, compact: bool = False) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        if compact:
            json.dump(payload, tmp, separators=(",", ":"))
        else:
            json.dump(payload, tmp, indent=indent)
            tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.rename(path)


def _extract_npz_metadata(npz_path: Path) -> dict | None:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if PACKED_METADATA_KEY not in data:
                return None
            payload = np.asarray(data[PACKED_METADATA_KEY], dtype=np.uint8).tobytes()
        return json.loads(payload.decode("utf-8"))
    except Exception as exc:
        print(f"[warn] failed to extract metadata from {npz_path}: {exc}", file=sys.stderr)
        return None


def write_rich_index(output_root: Path) -> int:
    """Write the rich_v1 index.json: shard-relative NPZ paths with the per-episode
    metadata blob inlined (so the loader can skip per-NPZ I/O at discovery time).
    Same schema/format as okto's ``build_packed_rich_index``."""
    npz_paths = sorted(output_root.glob("shard_*/*.npz"))
    episodes = []
    for p in npz_paths:
        meta = _extract_npz_metadata(p)
        if meta is None:
            continue
        episodes.append({"path": str(p.relative_to(output_root)), "metadata": meta})
    episodes.sort(key=lambda ep: ep["path"])
    _write_json_atomic(
        output_root / "index.json",
        {"format": "rich_v1", "num_episodes": len(episodes), "episodes": episodes},
        compact=True,
    )
    return len(episodes)


def write_format_manifest(output_root: Path, args: argparse.Namespace, total_episodes: int) -> None:
    from datetime import datetime, timezone

    _write_json_atomic(
        output_root / "format_manifest.json",
        {
            "format": "okto_packed",
            "version": 1,
            "dataset_type": args.dataset_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_roots": [str(args.source_root.resolve())],
            "output_root": str(output_root),
            "rgb_codec": args.rgb_codec,
            "rgb_quality": int(args.rgb_quality),
            "flow_backend": FLOW_BACKEND,
            "flow_codec": args.flow_codec,
            "flow_resolution": list(args.flow_resolution) if args.flow_resolution else None,
            "cuda_empty_cache_policy": args.cuda_empty_cache_policy,
            "io_mode": "baseline",
            "motion_codec": "zstd_npz",
            "skip_flow": bool(args.skip_flow),
            "skip_motion_tracks": True,
            "total_episodes": total_episodes,
            "views": args.views,
        },
        indent=2,
    )


def _write_progress(progress_path: Path, *, completed: int, total: int, last_episode: str):
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "completed": completed,
        "total": total,
        "last_episode": last_episode,
        "timestamp": time.time(),
    }) + "\n")
    tmp.rename(progress_path)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack PushT episodes (gym-pusht renders + trajectory source) into "
        "VERA's packed-NPZ training format with MegaFlow optical flow.",
    )
    parser.add_argument("--dataset-type", type=str, default="pusht",
                        choices=("pusht", "pusht_noise"),
                        help="'pusht' (default): human-teleop episodes, trajectory from the "
                             "replay zarr (--pusht-zarr-root). 'pusht_noise': DINO-WM "
                             "random-exploration episodes, trajectory from "
                             "--pusht-noise-source/train/states.pth.")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root of the rendered episodes (<root>/<episode_id>/00_rgb.mp4).")
    parser.add_argument("--pusht-zarr-root", type=Path, default=None,
                        help="Dir containing pusht_cchi_v7_replay.zarr (or the .zarr itself). "
                             "Required for --dataset-type pusht.")
    parser.add_argument("--pusht-noise-source", type=Path, default=None,
                        help="DINO-WM PushT dataset root (the dir whose train/ contains "
                             "states.pth). Required for --dataset-type pusht_noise.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=-1,
                        help="Number of episodes to process (-1 = all from start-index).")
    parser.add_argument("--views", type=str, default="00",
                        help="Comma-separated camera view names (released pack: '00').")

    # Codec / flow options (defaults match the released pusht-packed dataset).
    parser.add_argument("--rgb-codec", type=str, default=DEFAULT_RGB_CODEC)
    parser.add_argument("--rgb-quality", type=int, default=DEFAULT_RGB_QUALITY)
    parser.add_argument("--flow-codec", type=str, default="qint8_zstd_npz")
    parser.add_argument("--flow-resolution", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Resize video to this resolution before computing flow "
                             "(omitted for the released pack: flow at native 504x504).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--megaflow-window-size", type=int, default=4)
    parser.add_argument("--megaflow-iters", type=int, default=8)
    parser.add_argument("--cuda-empty-cache-policy", type=str, default="episode",
                        choices=("always", "episode", "never"))
    parser.add_argument("--temporal-stride", type=int, default=1,
                        help="Subsample every Nth source frame (1 for the released pack).")
    parser.add_argument("--progress-log-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--skip-flow", action="store_true", default=False,
                        help="Pack RGB + trajectory only (no MegaFlow / GPU needed).")
    parser.add_argument("--skip-index", action="store_true", default=False,
                        help="Skip (re)writing index.json + format_manifest.json at the end "
                             "(useful for sharded multi-job runs; run once at the end without it).")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    views = [v.strip() for v in args.views.split(",")]

    if args.dataset_type == "pusht_noise":
        if args.pusht_noise_source is None:
            parser.error("--dataset-type pusht_noise requires --pusht-noise-source")
        adapter = PushTNoiseAdapter(args)
    else:
        if args.pusht_zarr_root is None:
            parser.error("--dataset-type pusht requires --pusht-zarr-root")
        adapter = PushTAdapter(args)
    episode_entries = adapter.discover_episodes(source_root)
    if not episode_entries:
        raise SystemExit(f"No episodes found under {source_root}")

    start = args.start_index
    count = args.num_episodes if args.num_episodes > 0 else (len(episode_entries) - start)
    selected = episode_entries[start : start + count]
    if not selected:
        raise SystemExit("No episodes selected after applying start-index/num-episodes.")

    print(f"[init] source_root={source_root}")
    print(f"[init] {len(selected)} episodes selected (global indices {start}..{start + len(selected) - 1})")
    print(f"[init] output_root={output_root}")
    flow_resolution = tuple(args.flow_resolution) if args.flow_resolution else None
    print(f"[init] views={views} flow_backend={FLOW_BACKEND} flow_codec={args.flow_codec} "
          f"flow_resolution={flow_resolution}")

    device = torch.device(args.device)
    megaflow_model = None if args.skip_flow else _load_megaflow_model(device)

    start_time = time.perf_counter()
    num_attempted = 0
    num_written = 0
    num_skipped = 0
    progress_path = output_root / ".progress.json"

    for local_idx, entry in enumerate(selected):
        if _SHUTDOWN_REQUESTED:
            print(f"[shutdown] Stopping after {num_written} written, {num_skipped} skipped")
            break

        num_attempted += 1
        global_index = start + local_idx
        rel_path = entry["relative_path"]
        episode_id = entry.get("episode_id", rel_path)
        ep_dir = source_root / rel_path

        if not ep_dir.is_dir():
            print(f"[skip] missing episode dir: {ep_dir}")
            continue

        output_path = output_root / shard_relative_npz_path(
            episode_index=global_index,
            episode_id=episode_id,
        )
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {episode_id}: already exists at {output_path}")
            num_skipped += 1
            continue

        videos = adapter.get_videos(ep_dir, views)
        if not videos:
            print(f"[skip] {episode_id}: no RGB videos found for views {views}")
            continue

        try:
            trajectory_arrays = adapter.load_trajectory(ep_dir, episode_id)
        except Exception as exc:
            print(f"[error] {episode_id}: failed loading trajectory: {exc}")
            continue

        num_frames = adapter.get_num_frames_from_trajectory(trajectory_arrays)

        tstride = int(args.temporal_stride)
        if tstride > 1:
            trajectory_arrays = {
                k: v[::tstride] if v.ndim >= 1 and v.shape[0] == num_frames else v
                for k, v in trajectory_arrays.items()
            }
            num_frames = len(next(iter(trajectory_arrays.values())))

        metadata = _build_base_metadata(
            episode_id=episode_id,
            source_relative_path=rel_path,
            num_frames=int(num_frames),
            views=list(videos.keys()),
            trajectory_arrays=trajectory_arrays,
            rgb_codec=args.rgb_codec,
            rgb_quality=int(args.rgb_quality),
        )
        # Note set matches the released pusht-packed / pusht-noise-packed episodes.
        # Deliberately no absolute source paths in the notes — the NPZs stay
        # machine-independent.
        metadata.notes.append(f"flow_backend:{FLOW_BACKEND}")
        metadata.notes.append(f"dataset_type:{args.dataset_type}")
        if tstride > 1:
            metadata.notes.append(f"temporal_stride:{tstride}")

        try:
            entries = _episode_entries(
                videos=videos,
                trajectory_arrays=trajectory_arrays,
                metadata=metadata,
                device=device,
                megaflow_model=megaflow_model,
                megaflow_window_size=int(args.megaflow_window_size),
                megaflow_iters=int(args.megaflow_iters),
                skip_flow=bool(args.skip_flow),
                flow_codec=args.flow_codec,
                flow_resolution=flow_resolution,
                rgb_codec=args.rgb_codec,
                rgb_quality=int(args.rgb_quality),
                cache_policy=args.cuda_empty_cache_policy,
                temporal_stride=tstride,
            )
            write_npz_stored_atomic(output_path, entries)
            print(f"[ok] wrote {output_path}")
            num_written += 1
        except Exception as exc:
            print(f"[error] failed packing {episode_id}: {type(exc).__name__}: {exc}")
            raise
        finally:
            _maybe_empty_cache(
                device,
                cache_policy=args.cuda_empty_cache_policy,
                stage="episode",
            )
            log_every = max(1, int(args.progress_log_every))
            if (local_idx + 1) % log_every == 0:
                elapsed = max(1e-9, time.perf_counter() - start_time)
                sec_per_episode = elapsed / max(1, num_attempted)
                print(
                    "[progress] "
                    f"{local_idx + 1}/{len(selected)} attempted, "
                    f"{num_written} written, {num_skipped} skipped, "
                    f"elapsed={elapsed:.1f}s, "
                    f"sec_per_episode={sec_per_episode:.2f}"
                )
                _write_progress(
                    progress_path,
                    completed=num_written + num_skipped,
                    total=len(selected),
                    last_episode=episode_id,
                )

    elapsed = time.perf_counter() - start_time
    print(
        f"[done] {num_written} written, {num_skipped} skipped, "
        f"{num_attempted} attempted in {elapsed:.1f}s"
    )

    if not args.skip_index:
        n_indexed = write_rich_index(output_root)
        write_format_manifest(output_root, args, total_episodes=n_indexed)
        print(f"[index] rich_v1 index.json ({n_indexed} episodes) + format_manifest.json written")


if __name__ == "__main__":
    main()
