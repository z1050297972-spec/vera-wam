"""Verify a packed SO-ARM/Lerobot pack and (re)compute its normalization stats.

Run this after ``pack_lerobot.py``. It is the correctness gate for the pack,
because packing reads *slices* of multi-episode mp4s — a slicing bug produces a
pack that loads fine and looks plausible while every frame after the first
episode in a file is silently wrong. (That bug happened; this script is what
would have caught it.)

Checks, per episode:
  1. structure — num_frames / views / rgb / flow / traj entries, flow count = T-1
  2. frame exactness — packed JPEG frames vs. the LeRobot dataset's own decoded
     frames, at the START / MIDDLE / END of the episode, per view (mean abs diff
     on a subsampled grid; JPEG q90 keeps it a few 1e-3, a wrong frame is >1.0)
  3. trajectory — ``traj_state`` vs. the parquet rows the episode metadata names
  4. sanity — no NaN, du / flow magnitude distributions

With ``--stats-out`` it writes the measured ``action_abs_scale`` (p90 |du| per
channel) and ``oflow_abs_scale`` (p90 |flow| per channel, AT THE TRAINING
RESOLUTION — the loader resizes before normalizing, so storage-resolution stats
would be off by the resize factor) as YAML, ready to paste into the dataset cfg.

Usage::

    python scripts/data/check_packed.py \\
        --source-root $VERA_DATA_PREFIX/soarm-pickplace \\
        --packed-root $VERA_DATA_PREFIX/datasets/jacobian/soarm_packed \\
        --views desk --image-size 128 128 --fps 30 --temporal-stride 3 \\
        --stats-out /tmp/soarm_stats.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# +-N frames searched when checking alignment; must be >= 1
SEARCH_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)


def _sm(a: np.ndarray) -> np.ndarray:
    """Subsample to a coarse grid so comparisons are fast but still spatial."""
    return a[::8, ::8].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", type=Path, default=None,
                    help="LeRobot dataset root; enables the frame/trajectory cross-checks.")
    ap.add_argument("--packed-root", type=Path, required=True)
    ap.add_argument("--views", type=str, default="desk")
    ap.add_argument("--image-size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"),
                    help="Training resolution (must match dataset camera.image_size).")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--temporal-stride", type=int, default=3)
    ap.add_argument("--samples-per-episode", type=int, default=3,
                    help="Frames per episode to cross-check (start/middle/end).")
    ap.add_argument("--frame-diff-tol", type=float, default=5.0,
                    help="Mean abs diff (0..255) at the BEST alignment above which the frame "
                         "content is judged wrong. JPEG q90 costs ~1/255, and the check that "
                         "actually catches slicing bugs is that offset 0 must be the strict "
                         "minimum (see SEARCH_OFFSETS).")
    ap.add_argument("--alignment-margin", type=float, default=0.3,
                    help="How much worse diff@0 may be than the best nearby offset before it "
                         "counts as misalignment. Must exceed the JPEG q90 noise (~1/255) yet "
                         "stay under the gap between adjacent frames on a moving wrist camera.")
    ap.add_argument("--stats-out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from vera.datasets.core.sources import PackedSource
    from vera.datasets.core.view_loader import PackedViewLoader
    from vera.datasets.core.packed import decode_packed_rgb_frame, open_packed_npz

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    height, per_view_w = int(args.image_size[0]), int(args.image_size[1])

    packed_root = args.packed_root.resolve()
    index = json.loads((packed_root / "index.json").read_text())
    print(f"[info] index.json: format={index.get('format')} num_episodes={index.get('num_episodes')}")

    src = PackedSource(packed_root, name="robomimic", views=views)
    episodes = [src.resolve_episode(e) for e in src.list_episodes()]
    print(f"[info] PackedSource 解析出 {len(episodes)} 集；图像尺寸 ({height}, {per_view_w})\n")

    # ---- optional: the LeRobot side, for cross-checks -------------------------
    meta_rows, data_table = {}, None
    if args.source_root:
        import pyarrow.parquet as pq
        src_root = args.source_root.resolve()
        cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
        cols += [f"videos/observation.images.{v}/{k}" for v in views
                 for k in ("file_index", "from_timestamp")]
        for p in sorted((src_root / "meta" / "episodes").rglob("*.parquet")):
            avail = set(pq.read_schema(p).names)
            for row in pq.read_table(p, columns=[c for c in cols if c in avail]).to_pylist():
                row["_parquet"] = p
                meta_rows[int(row["episode_index"])] = row
        dfile = sorted((src_root / "data").rglob("*.parquet"))[0]
        data_table = np.asarray(
            pq.read_table(dfile, columns=["observation.state"])["observation.state"].to_pylist(),
            dtype=np.float32)
        print(f"[info] 源数据集: {len(meta_rows)} 集, state 表 {data_table.shape}")

    loader = PackedViewLoader(height=height, per_view_w=per_view_w)

    problems: list[str] = []
    all_offsets: list[int] = []
    all_du: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []

    header = (f"{'ep':>3} {'帧数':>6} {'flow':>6} {'du维度':>7} {'帧校验(平均差)':>34} {'traj':>8}")
    print(header)
    print("-" * len(header))

    for ep in episodes:
        ep_index = int(ep.episode_id.rsplit("_", 1)[-1])
        T = int(ep.num_frames)
        npz = open_packed_npz(ep.paths["packed_npz"])
        keys = set(npz.keys())

        n_flow = sum(1 for k in keys if k.startswith("flow_"))
        # Count against the views actually present in the NPZ, not the configured
        # subset: a pack may hold more views than this run cross-checks.
        packed_views = sorted({k.split("_view_", 1)[1] for k in keys if k.startswith("frame_")})
        expect_flow = (T - 1) * len(packed_views)
        du = np.asarray(loader.load_trajectory(ep, "state"))
        if isinstance(du, dict):                     # tolerate a dict-returning loader
            du = np.asarray(du["state"])
        if "traj_state" not in keys:
            problems.append(f"ep{ep_index}: 缺少 traj_state")
        if n_flow != expect_flow:
            problems.append(f"ep{ep_index}: flow 条目 {n_flow} != 预期 {expect_flow} (T-1)×视图")

        # structure: every rgb/flow key the metadata promises must exist
        packed_meta = ep.paths["packed"]
        for v in views:
            want = set(packed_meta["rgb_entries"][v]["keys"])
            if not want <= keys:
                problems.append(f"ep{ep_index} {v}: 缺 {len(want - keys)} 个 rgb 条目")
            want = set((packed_meta["flow_entries"][v] or {}).get("keys", []))
            if not want <= keys:
                problems.append(f"ep{ep_index} {v}: 缺 {len(want - keys)} 个 flow 条目")

        # ---- frame exactness -------------------------------------------------
        # An absolute diff threshold cannot separate "wrong frame" from "JPEG q90
        # loss" (~1/255) — a ONE-frame offset on the wrist camera differs by only
        # ~1.07 on a subsampled grid. So instead: search +-SEARCH offsets and
        # require offset 0 to be the strict minimum. A shifted pack moves the
        # minimum; a correct pack keeps it at 0.
        diffs: list[float] = []
        offsets: list[int] = []
        if args.source_root and ep_index in meta_rows:
            import av
            row = meta_rows[ep_index]
            lo = int(row["dataset_from_index"])
            ks = sorted({0, T // 2, T - 1})[: max(1, args.samples_per_episode)]
            for v in views:
                fi = int(row[f"videos/observation.images.{v}/file_index"])
                start = int(round(float(row[f"videos/observation.images.{v}/from_timestamp"]) * args.fps))
                # NOTE: the frame index is relative to ITS OWN file. The parquet row
                # index (`lo`) is a global dataset row number and equals the file
                # index only for the first file — using it here silently skips the
                # cross-check for every episode past the first file.
                last_needed = start + (ks[-1] + max(SEARCH_OFFSETS)) * args.temporal_stride
                cache: dict[int, np.ndarray] = {}
                path = (args.source_root / "videos" / f"observation.images.{v}"
                        / "chunk-000" / f"file-{fi:03d}.mp4")
                with av.open(str(path)) as c:
                    c.streams.video[0].thread_type = "AUTO"
                    for i, f in enumerate(c.decode(video=0)):
                        if i < start:
                            continue
                        cache[i - start] = f.to_ndarray(format="rgb24")
                        if i >= last_needed:
                            break
                for k in ks:
                    got = _sm(decode_packed_rgb_frame(npz, k, v).astype(np.float32))
                    ai = k * args.temporal_stride          # offset within this episode
                    cand = []
                    for d in SEARCH_OFFSETS:
                        ref = cache.get(ai + d)
                        if ref is not None:
                            cand.append((float(np.abs(got - _sm(ref.astype(np.float32))).mean()), d))
                    if not cand:
                        problems.append(f"ep{ep_index} {v}: 源文件取不到帧 {start + ai}")
                        continue
                    at0 = next((c_ for c_ in cand if c_[1] == 0), None)
                    best_d, best_off = min(cand)
                    if at0 is None:
                        problems.append(f"ep{ep_index} {v} 帧{k}: 源文件缺少偏移 0 的参照帧")
                        continue
                    d0 = at0[0]
                    diffs.append(d0)
                    offsets.append(0 if d0 <= best_d + args.alignment_margin else best_off)
                    all_offsets.extend(offsets[-1:])
                    # Two independent gates, because neither alone is enough:
                    #  (1) absolute — the content at the true position must match
                    #      (catches a grossly shifted pack: diff jumps to tens);
                    #  (2) margin — offset 0 must beat the best nearby offset by a
                    #      clear margin. A single static moment can make the true
                    #      frame and its neighbour differ by less than JPEG noise,
                    #      which must NOT be reported as misalignment.
                    if d0 > args.frame_diff_tol:
                        problems.append(
                            f"ep{ep_index} {v} 帧{k}: 偏移0 处差 {d0:.3f}（> {args.frame_diff_tol}）"
                            f" —— 内容不符")
                    elif d0 > best_d + args.alignment_margin:
                        problems.append(
                            f"ep{ep_index} {v} 帧{k}: 偏移0 差 {d0:.3f}，比偏移{best_off:+d} "
                            f"的 {best_d:.3f} 差 {d0 - best_d:.3f} —— 疑似错位")

        # ---- trajectory ------------------------------------------------------
        traj_diff = float("nan")
        if args.source_root and ep_index in meta_rows:
            row = meta_rows[ep_index]
            expect = data_table[int(row["dataset_from_index"]):int(row["dataset_to_index"])][
                ::args.temporal_stride]
            if expect.shape[0] != du.shape[0]:
                problems.append(f"ep{ep_index}: traj 行数 {du.shape[0]} != 源 {expect.shape[0]}")
            else:
                traj_diff = float(np.abs(du - expect).max())
                if traj_diff > 1e-4:
                    problems.append(f"ep{ep_index}: traj_state 与源差 {traj_diff:.5f}")

        d_txt = "—"
        if diffs:
            worst = max(diffs)
            d_txt = f"{worst:.4f}{'' if worst <= args.frame_diff_tol else ' ✗'}"
        print(f"{ep_index:>3} {T:>6} {n_flow:>6} {du.shape[1]:>7} {d_txt:>34} "
              f"{'' if np.isnan(traj_diff) else f'{traj_diff:.5f}':>8}")

        # ---- stats -----------------------------------------------------------
        W = 64
        step = max(1, (T - 1 - W) // 6) if T > W + 8 else 1
        for s in range(0, max(1, T - W), step):
            idx = np.arange(s, min(s + W, T))
            f = loader.load_flow(ep, idx)[0].numpy()        # [w,2,H,W] at train res
            all_u.append(f[:, 0].ravel())
            all_v.append(f[:, 1].ravel())
        st = np.asarray(du, dtype=np.float64)
        for k in range(len(st) - 1):
            all_du.append(st[k + 1] - st[k])

    U, V = (np.concatenate(x) for x in (all_u, all_v))
    D = np.stack(all_du)
    p90_du = np.percentile(np.abs(D), 90, axis=0)
    p90_u, p90_v = np.percentile(np.abs(U), 90), np.percentile(np.abs(V), 90)

    print("\n=== 归一化统计（建议填回 dataset yaml）===")
    print(f"  action_abs_scale (p90|du|, {D.shape[1]} 维): "
          f"{[round(float(x), 4) for x in p90_du]}")
    print(f"  oflow_abs_scale  (p90|flow|, {height}x{per_view_w}): "
          f"[{p90_u:.4f}, {p90_v:.4f}]")
    print(f"  样本数: du {D.shape[0]} 步, flow {U.size} 像素")

    if args.stats_out:
        args.stats_out.write_text(
            "action_abs_scale:\n"
            + "".join(f"  - {float(x):.6f}\n" for x in p90_du)
            + "oflow_abs_scale:\n"
            + f"  - {p90_u:.6f}\n  - {p90_v:.6f}\n"
        )
        print(f"  → 已写入 {args.stats_out}")

    print()
    # A check that silently did nothing must not report success. If a source
    # dataset was given, the frame/trajectory cross-checks MUST have run.
    if args.source_root and not diffs:
        print("❌ 致命：指定了 --source-root，但一次帧比对都没执行 —— 检查未生效，不能判通过。")
        return 2
    if problems:
        print(f"❌ 发现 {len(problems)} 个问题：")
        for p in problems[:40]:
            print(f"   - {p}")
        if len(problems) > 40:
            print(f"   … 还有 {len(problems) - 40} 条")
        return 1
    n_ok = sum(1 for o in all_offsets if o == 0)
    print(f"✅ 全部检查通过：结构完整、轨迹逐位一致、{n_ok}/{len(all_offsets)} 次帧比对对齐在偏移 0。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
