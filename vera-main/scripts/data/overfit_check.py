"""Overfit sanity check for a packed dataset + the Jacobian IDM.

Answers one question: **can this data actually drive this model to a low loss?**
It builds the real dataset and the real algorithm through VERA's own registries,
takes a few fixed batches, and runs plain gradient descent on them, printing the
loss trajectory. A correct pipeline drives the training loss toward ~0; if it
plateaus high, the fault is in the data (wrong normalization, misaligned frames,
a view with no signal) or the model wiring — not in the training loop.

This is a *wiring* test, not a benchmark: it says nothing about generalization.

Usage::

    python scripts/data/overfit_check.py \\
        --config-name config_jacobian_soarm_vggt \\
        --batches 2 --steps 150 --lr 1e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-name", default="config_jacobian_soarm_vggt")
    ap.add_argument("--batches", type=int, default=2, help="Fixed batches to cycle over.")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--final-loss-threshold", type=float, default=None,
                    help="Exit non-zero if the final loss is above this. By default only "
                         "the trajectory is reported (loss scale is config-dependent).")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from vera.datasets.registry import build_dataset
    from vera.idm.jacobian.image_jacobian import ImageJacobian  # noqa: F401  (registers)
    from vera.idm.registry import resolve_algorithm_cfg, resolve_algorithm_instance
    import vera.idm.jacobian.models.vggt_jacobian_field  # noqa: F401  (registers model)

    cfg_dir = Path(__file__).resolve().parents[2] / "vera" / "configurations"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name=args.config_name)

    ds = build_dataset(cfg.dataset, stage="training")
    print(f"[data] {type(ds).__name__}: {len(ds._episodes)} 集, "
          f"action_model={type(ds.action_model).__name__}, "
          f"图像=({ds.cfg.height},{ds.cfg.per_view_w}), layout={ds.cfg.layout}")

    device = torch.device(args.device)
    batches = []
    g = torch.Generator().manual_seed(0)
    for _ in range(args.batches):
        s = ds[int(torch.randint(0, len(ds._episodes), (1,), generator=g))]
        batches.append({
            "rgb": s["rgb"].unsqueeze(0).to(device),
            "flow": s["flow"].unsqueeze(0).to(device),
            "du": s["du"].unsqueeze(0).to(device),
        })
        print(f"[data] 固定一批: rgb{tuple(batches[-1]['rgb'].shape)} "
              f"flow{tuple(batches[-1]['flow'].shape)} du{tuple(batches[-1]['du'].shape)} "
              f"|du|p90={s['du'].abs().flatten().kthvalue(int(0.9*s['du'].numel())).values:.2f} "
              f"|flow|p90={s['flow'].abs().flatten().kthvalue(int(0.9*s['flow'].numel())).values:.2f}")

    algo_cfg = resolve_algorithm_cfg(cfg.algorithm)
    algo = resolve_algorithm_instance(algo_cfg)
    # training_step ends with a throughput-logging helper that reads
    # `self.trainer.world_size`; standalone there is no Trainer, and Lightning
    # raises on that access. Neutralize just that logging call — the loss path
    # itself is untouched.
    algo._log_training_speed = lambda *a, **k: None
    algo = algo.to(device)
    trainable = [p for p in algo.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in algo.parameters())
    print(f"[model] {type(algo).__name__}: 可训练 {n_tr/1e6:.1f}M / 总 {n_all/1e6:.1f}M 参数")

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"[optim] AdamW lr={args.lr}\n")
    print(f"{'step':>6} {'total':>12} {'flow':>12} {'du':>12}  说明")

    losses = []
    for step in range(args.steps):
        b = batches[step % len(batches)]
        opt.zero_grad(set_to_none=True)
        out = algo.training_step(b, 0)
        loss = _total_loss(out)
        loss.backward()
        opt.step()
        val = float(loss.detach())
        losses.append(val)
        if step % args.log_every == 0 or step == args.steps - 1:
            parts = {k: float(v) for k, v in _parts(out).items()}
            note = ""
            if step and abs(val - losses[max(0, step - args.log_every)]) < 1e-6:
                note = "（与上次相同）"
            print(f"{step:>6} {val:>12.5f} {parts.get('flow', float('nan')):>12.5f} "
                  f"{parts.get('du', float('nan')):>12.5f}  {note}")

    first = sum(losses[: args.log_every]) / args.log_every
    last = sum(losses[-args.log_every:]) / args.log_every
    print(f"\n前 {args.log_every} 步均值 {first:.5f} → 后 {args.log_every} 步均值 {last:.5f}"
          f"（下降 {(1 - last/first)*100 if first else 0:.1f}%）")

    if args.final_loss_threshold is not None and last > args.final_loss_threshold:
        print(f"❌ 末段损失 {last:.5f} > 阈值 {args.final_loss_threshold}")
        return 1
    if not (last < first):
        print("❌ 损失没有下降 —— 数据或模型接线有问题")
        return 1
    print("✅ 损失下降：数据 → 模型 → 损失 → 参数更新 链路正常")
    return 0


def _total_loss(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in ("loss", "total", "training/total"):
            if k in out:
                return out[k]
        for v in out.values():
            if torch.is_tensor(v) and v.requires_grad:
                return v
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v) and v.requires_grad:
                return v
    raise RuntimeError(f"无法从算法输出里取到 loss: {type(out)}")


def _parts(out) -> dict:
    if isinstance(out, dict):
        return {k: v for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1}
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
