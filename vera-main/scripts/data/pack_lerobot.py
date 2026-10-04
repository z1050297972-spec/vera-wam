"""Pack a LeRobot video dataset into VERA's packed-NPZ training format.

Same output format as ``pack_pusht.py`` (``okto_packed`` v1: one NPZ per episode
holding JPEG frames, MegaFlow optical flow, the trajectory arrays and a metadata
blob, plus ``index.json`` + ``format_manifest.json``), but the source is a
LeRobot dataset instead of per-episode mp4 directories.

The one structural difference that matters: **a LeRobot mp4 holds many
concatenated episodes**, not one. ``meta/episodes/<chunk>/<file>.parquet`` gives,
per episode and per view, which mp4 the episode lives in and its time range
inside that file::

    videos/observation.images.desk/file_index       -> which file-00X.mp4
    videos/observation.images.desk/from_timestamp   -> where it starts in THAT file

So frames are decoded **by absolute frame range, sequentially**, with one open
container per file kept across episodes (episodes are packed in order, so each
file is decoded exactly once). Decoding a whole file instead is not an option:
the first ``file-000.mp4`` here is 8,942 frames of 1280x720, i.e. ~25 GB as uint8.

Decoding is verified pixel-exact against LeRobot's own reader for every episode
and view (see the docstring note in ``LerobotVideoReader``).

Trajectory: ``observation.state`` from ``data/<chunk>/<file>.parquet``, sliced
with the episode's ``dataset_from_index``/``dataset_to_index``, becomes
``traj_state`` [T, D] float32 — which is what ``action_mode: qpos_delta`` reads to
build the per-step action delta ``du``. Global per-channel bounds go in as
``traj_q_min`` / ``traj_q_max``.

Usage::

    python scripts/data/pack_lerobot.py \\
        --source-root ../../data/soarm-pickplace \\
        --output-root $VERA_DATA_PREFIX/datasets/jacobian/soarm_packed \\
        --views desk,wide \\
        --temporal-stride 3 \\
        --flow-resolution 256 256

``--temporal-stride`` subsamples 30 fps -> 10 fps. Do it **before** the flow is
computed (this script always does) so optical flow and ``du`` describe the same
time step; the published IDM configs assume ~10 fps-scale deltas.

Requires MegaFlow + a CUDA GPU unless ``--skip-flow`` (RGB + trajectory only,
which the IDM cannot train on — it hardcodes ``load_flow=True``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pack_pusht import (  # noqa: E402  (path set up above)
    _build_base_metadata,
    _episode_entries,
    _load_megaflow_model,
    _maybe_empty_cache,
    _SHUTDOWN_REQUESTED,
    write_format_manifest,
    write_rich_index,
    _write_progress,
)
from vera.utils.droid_packed_format import (  # noqa: E402
    DEFAULT_RGB_CODEC,
    DEFAULT_RGB_QUALITY,
    shard_relative_npz_path,
    write_npz_stored_atomic,
)

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


def full_view_key(view: str) -> str:
    """``desk`` -> ``observation.images.desk`` (LeRobot's parquet/video key)."""
    return view if view.startswith("observation.images.") else f"observation.images.{view}"


# ============================================================================
# LeRobot source adapter
# ============================================================================


class LerobotVideoReader:
    """Sequential frame-range decoder over LeRobot's multi-episode mp4s.

    One container is kept open per (view, file) and advanced forward, because
    episodes are packed in ascending order and each episode's slice starts where
    the previous one ended. Frame indices are **absolute within the file**, not
    within the episode — that is the whole point, since a file holds several
    episodes.
    """

    def __init__(self, source_root: Path, fps: float):
        self.root = source_root
        self.fps = fps
        self._streams: dict[tuple[str, int], tuple[object, object, int]] = {}

    def _open(self, view: str, file_index: int):
        key = (view, file_index)
        entry = self._streams.get(key)
        if entry is None:
            import av

            path = self.root / "videos" / full_view_key(view) / "chunk-000" / f"file-{file_index:03d}.mp4"
            container = av.open(str(path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            entry = (container, iter(container.decode(video=0)), 0)
            self._streams[key] = entry
        return entry

    def read(self, view: str, file_index: int, start_frame: int, num_frames: int) -> np.ndarray:
        """Decode ``num_frames`` starting at absolute ``start_frame`` in that file."""
        key = (view, file_index)
        container, frames, pos = self._open(view, file_index)
        if pos > start_frame:
            # Episodes packed out of order (or a restarted run): re-open rather than
            # silently return frames from the wrong position.
            container.close()
            del self._streams[key]
            container, frames, pos = self._open(view, file_index)
        out: list[np.ndarray] = []
        for frame in frames:
            if pos < start_frame:
                pos += 1
                continue
            out.append(frame.to_ndarray(format="rgb24"))
            pos += 1
            if len(out) >= num_frames:
                break
        # The iterator has advanced, so the position must be written back — without
        # this the next episode restarts counting at 0 while the iterator is already
        # past `start_frame`, and silently reads the wrong frames.
        self._streams[key] = (container, frames, pos)
        if len(out) < num_frames:
            raise RuntimeError(
                f"{view} file-{file_index:03d}.mp4: 帧 {start_frame}..{start_frame + num_frames - 1} "
                f"只解出 {len(out)} 帧（该文件可能已到末尾）"
            )
        return np.stack(out)

    def close(self) -> None:
        for container, _frames, _pos in self._streams.values():
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass
        self._streams.clear()


class LerobotAdapter:
    """Reads episode list, trajectories and per-view frame ranges from a LeRobot dataset."""

    def __init__(self, source_root: Path, state_key: str, fps: float):
        if pq is None:
            raise SystemExit("需要 pyarrow：pip install pyarrow")
        self.root = source_root
        self.state_key = state_key
        self.fps = fps
        self.reader = LerobotVideoReader(source_root, fps)
        self._states: dict[tuple[int, int], list] = {}

        meta = json.loads((source_root / "meta" / "info.json").read_text())
        self.dataset_fps = float(meta.get("fps", fps))
        self.total_episodes = int(meta.get("total_episodes", 0))

    def _ep_table(self) -> list[Path]:
        files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            raise SystemExit(f"{self.root}/meta/episodes 下没有 parquet")
        return files

    def load_episodes(self, views: list[str]) -> list[dict]:
        """One dict per episode: index, length, parquet row range, per-view slice."""
        rows: list[dict] = []
        for path in self._ep_table():
            available = set(pq.read_schema(path).names)
            cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
            for view in views:
                key = full_view_key(view)
                cols += [f"videos/{key}/file_index", f"videos/{key}/from_timestamp"]
            cols = [c for c in cols if c in available]
            missing = [v for v in views
                       if f"videos/{full_view_key(v)}/file_index" not in available]
            if missing:
                raise SystemExit(f"{path} 缺少视图 {missing} 的视频列")
            for row in pq.read_table(path, columns=cols).to_pylist():
                row["_parquet"] = path
                rows.append(row)
        rows.sort(key=lambda r: r["episode_index"])
        return rows

    def load_states(self, data_parquet: Path) -> np.ndarray:
        """Full ``observation.state`` table as [N, D] float32 (cached per file)."""
        # Key on the PATH, not id(path): CPython reuses ids of collected objects,
        # so an id-keyed cache can silently return another file's states.
        key = str(data_parquet)
        if key not in self._states:
            table = pq.read_table(data_parquet, columns=[self.state_key])
            self._states[key] = np.asarray(table[self.state_key].to_pylist(), dtype=np.float32)
        return self._states[key]

    def load_trajectory(self, ep: dict) -> dict[str, np.ndarray]:
        # Read ONLY the two locating columns: the episodes table also carries ~110
        # nested per-episode stats columns, and pulling all of it per episode is
        # needlessly slow.
        table = pq.read_table(ep["_parquet"], columns=["data/file_index", "data/chunk_index"])
        file_index = int(table["data/file_index"][0].as_py())
        chunk_index = int(table["data/chunk_index"][0].as_py())
        path = self.root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        states = self.load_states(path)
        lo = int(ep["dataset_from_index"])
        hi = int(ep["dataset_to_index"])
        state = states[lo:hi]
        if state.shape[0] != int(ep["length"]):
            raise RuntimeError(
                f"ep{ep['episode_index']}: parquet 行数 {state.shape[0]} != length {ep['length']}"
            )
        return {
            "state": state,
            "q_min": states.min(axis=0).astype(np.float32),
            "q_max": states.max(axis=0).astype(np.float32),
        }

    def view_slice(self, ep: dict, view: str) -> tuple[int, int]:
        """(file_index, absolute start frame) for one episode/view."""
        key = full_view_key(view)
        file_index = int(ep[f"videos/{key}/file_index"])
        from_ts = float(ep[f"videos/{key}/from_timestamp"])
        return file_index, int(round(from_ts * self.fps))


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack a LeRobot video dataset (multi-episode mp4s + parquet) into "
        "VERA's packed-NPZ training format with MegaFlow optical flow.",
    )
    parser.add_argument("--source-root", type=Path, required=True,
                        help="LeRobot dataset root (contains meta/, data/, videos/).")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--views", type=str, default="desk,wide",
                        help="Comma-separated observation.images.* keys (default desk,wide).")
    parser.add_argument("--state-key", type=str, default="observation.state",
                        help="parquet column holding the joint state -> traj_state.")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Native fps of the recordings; converts from_timestamp to frame indices.")

    parser.add_argument("--rgb-codec", type=str, default=DEFAULT_RGB_CODEC)
    parser.add_argument("--rgb-quality", type=int, default=DEFAULT_RGB_QUALITY)
    parser.add_argument("--flow-codec", type=str, default="qint8_zstd_npz")
    parser.add_argument("--flow-resolution", type=int, nargs=2, default=[256, 256],
                        metavar=("H", "W"),
                        help="Video is resized to this before MegaFlow runs, and the flow is "
                             "stored at that size. 256 keeps detail while fitting 24 GB; 128 is "
                             "faster but loses fine motion. Omit for native-resolution flow "
                             "(needs a lot of GPU memory).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--megaflow-window-size", type=int, default=4)
    parser.add_argument("--megaflow-iters", type=int, default=8)
    parser.add_argument("--cuda-empty-cache-policy", type=str, default="episode",
                        choices=("always", "episode", "never"))
    parser.add_argument("--temporal-stride", type=int, default=3,
                        help="Keep every Nth source frame (default 3: 30 fps -> 10 fps, matching "
                             "the scale of the released configs' du / flow statistics).")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--progress-log-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--skip-flow", action="store_true", default=False,
                        help="RGB + trajectory only. NOTE: the IDM cannot train on this — it "
                             "hardcodes load_flow=True.")
    parser.add_argument("--skip-index", action="store_true", default=False)
    parser.add_argument("--dataset-type", type=str, default="soarm",
                        help="Only recorded in format_manifest.json.")
    args = parser.parse_args()

    import time

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    flow_resolution = tuple(args.flow_resolution) if args.flow_resolution else None
    tstride = max(1, int(args.temporal_stride))

    adapter = LerobotAdapter(source_root, args.state_key, float(args.fps))
    episodes = adapter.load_episodes(views)
    if not episodes:
        raise SystemExit(f"{source_root} 下没有 episode")

    start = max(0, args.start_index)
    count = args.num_episodes if args.num_episodes > 0 else (len(episodes) - start)
    selected = episodes[start : start + count]

    print(f"[init] source_root={source_root}")
    print(f"[init] {len(selected)} 集（全局序号 {start}..{start + len(selected) - 1}） / 共 {len(episodes)}")
    print(f"[init] output_root={output_root}")
    print(f"[init] views={views} state_key={args.state_key} temporal_stride={tstride} "
          f"(-> {adapter.dataset_fps / tstride:.1f} fps)")
    print(f"[init] flow_backend=megaflow codec={args.flow_codec} flow_resolution={flow_resolution}")

    device = torch.device(args.device)
    megaflow_model = None if args.skip_flow else _load_megaflow_model(device)

    t_start = time.perf_counter()
    num_written = num_skipped = 0
    progress_path = output_root / ".progress.json"

    try:
        for local_idx, ep in enumerate(selected):
            if _SHUTDOWN_REQUESTED:
                print(f"[shutdown] 停止：已写 {num_written}，跳过 {num_skipped}")
                break

            ep_id = f"episode_{int(ep['episode_index']):06d}"
            output_path = output_root / shard_relative_npz_path(
                episode_index=start + local_idx, episode_id=ep_id,
            )
            if output_path.exists() and not args.overwrite:
                print(f"[skip] {ep_id}: 已存在 {output_path}")
                num_skipped += 1
                continue

            length = int(ep["length"])
            trajectory = adapter.load_trajectory(ep)
            num_frames = length
            if tstride > 1:
                trajectory = {
                    k: (v[::tstride] if v.ndim >= 1 and v.shape[0] == num_frames else v)
                    for k, v in trajectory.items()
                }
                num_frames = int(trajectory["state"].shape[0])

            # Decode each view's slice (already strided) and keep only one view in
            # memory at a time: at 30->10 fps a 1,670-frame episode is ~1.5 GB per view.
            views_with_frames: dict[str, np.ndarray] = {}
            for view in views:
                file_index, start_frame = adapter.view_slice(ep, view)
                frames = adapter.reader.read(view, file_index, start_frame, length)
                if frames.shape[0] != length:
                    raise RuntimeError(
                        f"{ep_id} {view}: 解出 {frames.shape[0]} 帧，元数据说 {length}"
                    )
                views_with_frames[view] = frames[::tstride]

            metadata = _build_base_metadata(
                episode_id=ep_id,
                source_relative_path=f"episode_{int(ep['episode_index']):06d}",
                num_frames=int(num_frames),
                views=list(views_with_frames.keys()),
                trajectory_arrays=trajectory,
                rgb_codec=args.rgb_codec,
                rgb_quality=int(args.rgb_quality),
            )
            metadata.notes.append("flow_backend:megaflow")
            metadata.notes.append(f"dataset_type:{args.dataset_type}")
            metadata.notes.append(f"source:lerobot:{source_root.name}")
            if tstride > 1:
                metadata.notes.append(f"temporal_stride:{tstride}")

            try:
                entries = _episode_entries(
                    videos=views_with_frames,
                    trajectory_arrays=trajectory,
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
                    temporal_stride=1,   # frames are already strided
                )
                write_npz_stored_atomic(output_path, entries)
                size_mb = output_path.stat().st_size / 1e6
                print(f"[ok] {ep_id}: {num_frames} 帧 x {len(views_with_frames)} 视图 "
                      f"-> {output_path.name} ({size_mb:.1f} MB)")
                num_written += 1
            finally:
                views_with_frames.clear()
                _maybe_empty_cache(device, cache_policy=args.cuda_empty_cache_policy,
                                   stage="episode")

            if (local_idx + 1) % max(1, int(args.progress_log_every)) == 0:
                elapsed = max(1e-9, time.perf_counter() - t_start)
                print(f"[progress] {local_idx + 1}/{len(selected)}，已写 {num_written}，"
                      f"跳过 {num_skipped}，用时 {elapsed:.1f}s，"
                      f"每集 {elapsed / (local_idx + 1):.1f}s")
                _write_progress(progress_path, completed=num_written + num_skipped,
                                total=len(selected), last_episode=ep_id)
    finally:
        adapter.reader.close()

    print(f"[done] 写入 {num_written}，跳过 {num_skipped}，用时 {time.perf_counter() - t_start:.1f}s")

    if not args.skip_index:
        n = write_rich_index(output_root)
        write_format_manifest(output_root, args, total_episodes=n)
        print(f"[index] index.json（{n} 集）+ format_manifest.json 已写入")


if __name__ == "__main__":
    main()
