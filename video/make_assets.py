"""Produce the real tensors the IDM teaching video renders.

Runs the actual model code path on the packed SO-ARM dataset and saves into
``video/assets/``:

* ``loss_curve.json``  — 260 steps of overfitting two fixed batches
* ``jacobian.npy``     — the learned per-joint Jacobian field [6, 2, 128, 128]
* ``gt_flow.npy`` / ``pred_flow.npy`` — real MegaFlow flow vs the model's prediction
* ``rgb.npy`` / ``du.npy`` / ``confidence.npy`` — the frame, its normalised action,
  and the model's per-pixel confidence

Every heat map and flow field in the video comes from this run; nothing is drawn
by hand. Re-run it whenever the model or the data changes, then re-render::

    cd vera-main && HF_ENDPOINT=https://hf-mirror.com \\
      VERA_DATA_PREFIX=<data root> python ../video/make_assets.py
    python ../video/make_idm_video.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "vera-main"
sys.path.insert(0, str(REPO))

CONFIG = "config_jacobian_soarm_vggt"
STEPS = 260
LR = 1e-4
BATCHES = 2


def main() -> int:
    from hydra import compose, initialize_config_dir

    from vera.datasets.registry import build_dataset
    from vera.idm.jacobian.models.base import InputCommand, InputObservation
    from vera.idm.registry import resolve_algorithm_cfg, resolve_algorithm_instance

    import vera.idm.jacobian.image_jacobian  # noqa: F401  (registers algorithm)
    import vera.idm.jacobian.models.vggt_jacobian_field  # noqa: F401  (registers model)

    out = HERE / "assets"
    out.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(REPO / "vera" / "configurations"), version_base=None):
        cfg = compose(config_name=CONFIG)

    ds = build_dataset(cfg.dataset, stage="training")
    print(f"[data] {type(ds).__name__}: {len(ds._episodes)} 集, "
          f"action_model={type(ds.action_model).__name__}, "
          f"图像=({ds.cfg.height},{ds.cfg.per_view_w})", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(0)
    picks = [int(torch.randint(0, len(ds._episodes), (1,), generator=g)) for _ in range(BATCHES)]
    batches = [{k: ds[i][k].unsqueeze(0).to(device) for k in ("rgb", "flow", "du")} for i in picks]

    algo = resolve_algorithm_instance(resolve_algorithm_cfg(cfg.algorithm))
    # training_step ends with a throughput helper that needs a live Trainer.
    algo._log_training_speed = lambda *a, **k: None
    algo = algo.to(device)
    trainable = [p for p in algo.parameters() if p.requires_grad]
    print(f"[model] {type(algo).__name__}: 可训练 {sum(p.numel() for p in trainable)/1e6:.1f}M / "
          f"总 {sum(p.numel() for p in algo.parameters())/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(trainable, lr=LR)
    curve: list[float] = []
    for step in range(STEPS):
        b = batches[step % len(batches)]
        opt.zero_grad(set_to_none=True)
        o = algo.training_step(b, 0)
        o["loss"].backward()
        opt.step()
        curve.append(float(o["loss"].detach()))
        if step % 20 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}  loss {curve[-1]:+.4f}", flush=True)

    json.dump(curve, open(out / "loss_curve.json", "w"))
    print(f"[loss] 首10均 {np.mean(curve[:10]):+.4f} -> 末10均 {np.mean(curve[-10:]):+.4f}", flush=True)

    with torch.no_grad():
        mo = algo.model(
            InputObservation(rgb=batches[1]["rgb"][:, 0]),
            InputCommand(du=batches[1]["du"][:, 0]),
        )

    np.save(out / "jacobian.npy", mo.jacobian.detach().cpu().numpy())
    np.save(out / "pred_flow.npy", mo.optical_flow.detach().cpu().numpy())
    np.save(out / "confidence.npy",
            mo.flow_confidence.detach().cpu().numpy() if mo.flow_confidence is not None
            else np.zeros((1, 1, 128, 128), np.float32))
    np.save(out / "gt_flow.npy", batches[1]["flow"][0].cpu().numpy())
    np.save(out / "rgb.npy", batches[1]["rgb"][0].cpu().numpy())
    np.save(out / "du.npy", batches[1]["du"][0].cpu().numpy())
    print(f"[assets] 已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
