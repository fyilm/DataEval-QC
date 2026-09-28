"""温度标定（Temperature Scaling）与可靠性图。

关键纪律：**T 只允许在验证集上用 NLL 优化**。
测试集 / 冻结评测集只做最终报告，否则校准指标不可信 ——
这是校准实验最容易犯的错，也是面试最常被追问的点。

多标签场景下，温度缩放作用在 logits 上再过 sigmoid，
NLL 用所有 (样本, 类别) 的二元交叉熵之和。
"""
from __future__ import annotations

import sys
import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import CLASSES, FIGURES_DIR, write_json  # noqa: E402
from src.evaluate import (  # noqa: E402
    evaluate_logits,
    reliability_data,
    sigmoid,
)


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.clamp(min=1e-3)


def fit_temperature(
    val_logits: np.ndarray, val_targets: np.ndarray, max_iter: int = 200
) -> tuple[float, float]:
    """在验证集上用 NLL 优化 T。返回 (T, 优化后的 NLL)。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(np.asarray(val_logits), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(val_targets), dtype=torch.float32, device=device)

    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS([scaler.T], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(scaler(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        nll = float(nn.functional.binary_cross_entropy_with_logits(scaler(x), y).item())
    return float(scaler.T.item()), nll


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return np.asarray(logits) / max(1e-3, T)


def _plot_reliability(bins: list[dict[str, float]], out: pathlib.Path, title: str) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    # 中文字体：Windows 上用微软雅黑，避免 DejaVu Sans 缺字形
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    x, acc, cnt = [], [], []
    for b in bins:
        if b["count"] > 0:
            x.append((b["lo"] + b["hi"]) / 2)
            acc.append(b["acc"])
            cnt.append(b["count"])
    fig, ax = plt.subplots(figsize=(4.4, 4.2), dpi=110)
    ax.bar(x, acc, width=0.09, color="#5DCAA5", edgecolor="#0F6E56", linewidth=0.5, label="实际正确率")
    ax.bar(x, x, width=0.09, color="none", edgecolor="#888780", linewidth=0.6, label="理想校准")
    ax.plot([0, 1], [0, 1], color="#888780", linewidth=1.0, linestyle="--")
    ax.set_xlabel("预测置信度")
    ax.set_ylabel("实际正确率")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def calibration_report(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    test_logits: np.ndarray,
    test_targets: np.ndarray,
    tag: str = "main",
) -> dict[str, Any]:
    """完整校准流程：val 拟合 T -> test 报告前后对比 + 可靠性图。"""
    T, nll_after_val = fit_temperature(val_logits, val_targets)

    before = evaluate_logits(test_logits, test_targets)
    after = evaluate_logits(apply_temperature(test_logits, T), test_targets)

    fig_pre = FIGURES_DIR / f"reliability_{tag}_before.png"
    fig_post = FIGURES_DIR / f"reliability_{tag}_after.png"
    _plot_reliability(before["reliability_bins"], fig_pre, f"标定前  ECE={before['ece_multilabel']:.3f}")
    _plot_reliability(after["reliability_bins"], fig_post, f"标定后(T={T:.3f})  ECE={after['ece_multilabel']:.3f}")

    report = {
        "tag": tag,
        "temperature": round(T, 4),
        "val_nll_after_fit": round(nll_after_val, 5),
        "before": {"ece": before["ece_multilabel"], "brier": before["brier"], "macro_f1": before["macro_f1"]},
        "after": {"ece": after["ece_multilabel"], "brier": after["brier"], "macro_f1": after["macro_f1"]},
        "ece_reduction": round(before["ece_multilabel"] - after["ece_multilabel"], 4),
        "figures": {"before": str(fig_pre.relative_to(FIGURES_DIR.parents[1])), "after": str(fig_post.relative_to(FIGURES_DIR.parents[1]))},
    }
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="温度标定")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    from src.common import ROOT, get_device, load_manifest
    from src.dataset import QCDataset
    from src.train import build_model, collect_logits

    ck = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    device = get_device()
    model = build_model(ck["model_name"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    records = [r for r in load_manifest() if not r.get("dedup_removed", False)]
    val_ds = QCDataset([r for r in records if r["split"] == "val"], args.img_size, train=False)
    test_ds = QCDataset([r for r in records if r["split"] == "test"], args.img_size, train=False)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh)
    vl, vt = collect_logits(model, dl(val_ds, False), device)
    tl, tt = collect_logits(model, dl(test_ds, False), device)

    rep = calibration_report(vl, vt, tl, tt, tag=args.tag)
    write_json(rep, ROOT / "reports" / f"calibration_{args.tag}.json")
    print(
        f"[calibrate] T={rep['temperature']}\n"
        f"            ECE {rep['before']['ece']} -> {rep['after']['ece']}（降 {rep['ece_reduction']}）\n"
        f"            Brier {rep['before']['brier']} -> {rep['after']['brier']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
