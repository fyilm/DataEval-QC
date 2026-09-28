"""多标签分类训练。

模型按设备自动选择规模：CPU 跑得动的轻量模型，GPU 上自动放开。
支持 EfficientNet-B0 / ResNet18 / MobileNetV3-Small 三种骨干，
ImageNet 预训练权重从 download.pytorch.org 拉取（实测可达）。

消融实验里"合成占比"通过 dataset 的 sample_ratio / exclude_source 控制：
    ratio r -> 训练集中合成缺陷帧占比
    r = 0   -> 全部真实清晰帧（无正样本，退化对照组，验证合成数据的必要性）
    r = 1   -> 全部合成缺陷帧（无负样本，同样退化）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    NUM_CLASSES,
    ROOT,
    get_device,
    set_seed,
    suggest_num_workers,
    write_json,
)
from src.dataset import QCDataset  # noqa: E402

MODELS = ("efficientnet_b0", "resnet18", "mobilenet_v3_small")


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """构建骨干网络并把分类头换成 6 类 multi-hot 输出。"""
    from torchvision import models

    weights = "IMAGENET1K_V1" if pretrained else None
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif name == "resnet18":
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=weights)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"未知模型 {name}，可选: {MODELS}")
    return m


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """推理一轮，返回 (logits, targets)。校准和评测都吃这个输出。"""
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def benchmark_throughput(model: nn.Module, device: torch.device, img_size: int, batch_size: int, n: int = 64) -> dict[str, float]:
    """实测训练吞吐，用来给消融网格定档。"""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.BCEWithLogitsLoss()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, 2, (batch_size, NUM_CLASSES), device=device).float()

    for _ in range(3):  # 预热
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()

    t0 = time.perf_counter()
    steps = max(1, n // batch_size)
    for _ in range(steps):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
    dt = time.perf_counter() - t0
    return {"img_per_sec": round(batch_size * steps / dt, 1), "batch_size": batch_size, "img_size": img_size}


def train_one(
    model_name: str,
    train_ds: QCDataset,
    val_ds: QCDataset,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    img_size: int = 160,
    num_workers: int | None = None,
    seed: int = 42,
    out_dir: pathlib.Path | None = None,
    log_every: int = 0,
) -> dict[str, Any]:
    set_seed(seed)
    if num_workers is None:
        num_workers = suggest_num_workers(device)

    pw = train_ds.pos_weight.to(device)
    model = build_model(model_name, pretrained=True).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    dl_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), drop_last=len(train_ds) > batch_size, persistent_workers=num_workers > 0,
    )
    dl_val = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    best = {"epoch": -1, "val_f1": -1.0}
    history = []
    from src.evaluate import macro_f1_from_logits  # 局部导入避免循环依赖

    for ep in range(epochs):
        model.train()
        total_loss, nb = 0.0, 0
        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            nb += 1
        sched.step()
        train_loss = total_loss / max(1, nb)

        logits, targets = collect_logits(model, dl_val, device)
        val_f1 = macro_f1_from_logits(logits, targets)
        history.append({"epoch": ep, "train_loss": round(train_loss, 5), "val_macro_f1": round(val_f1, 5)})

        if log_every:
            print(f"    epoch {ep+1}/{epochs} loss={train_loss:.4f} val_macroF1={val_f1:.4f}", flush=True)

        if val_f1 > best["val_f1"]:
            best = {"epoch": ep, "val_f1": round(val_f1, 5)}
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "model_name": model_name, "epoch": ep}, out_dir / "best.pt")

    result: dict[str, Any] = {
        "model": model_name,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_f1"],
        "history": history,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "img_size": img_size,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
    }
    if out_dir:
        result["checkpoint"] = str((out_dir / "best.pt").relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练多标签质检分类器")
    ap.add_argument("--model", choices=list(MODELS), default="mobilenet_v3_small")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="每类只取 N 张，用于快速验证")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args(argv)

    from src.common import load_manifest

    device = get_device(args.device)
    records = load_manifest()
    active = [r for r in records if not r.get("dedup_removed", False)]
    train_recs = [r for r in active if r["split"] == "train"]
    val_recs = [r for r in active if r["split"] == "val"]

    if args.limit:
        rng = np.random.default_rng(args.seed)
        train_recs = [train_recs[i] for i in rng.choice(len(train_recs), min(args.limit, len(train_recs)), replace=False)]
        val_recs = [val_recs[i] for i in rng.choice(len(val_recs), min(args.limit // 2, len(val_recs)), replace=False)]

    print(f"[train] device={device} model={args.model} train={len(train_recs)} val={len(val_recs)}")
    train_ds = QCDataset(train_recs, args.img_size, train=True, seed=args.seed)
    val_ds = QCDataset(val_recs, args.img_size, train=False, seed=args.seed)

    res = train_one(
        args.model, train_ds, val_ds, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        img_size=args.img_size, seed=args.seed,
        out_dir=ROOT / args.out, log_every=1,
    )
    write_json(res, ROOT / "reports" / "train_last.json")
    print(f"[train] best_val_macroF1={res['best_val_macro_f1']:.4f} (epoch {res['best_epoch']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
