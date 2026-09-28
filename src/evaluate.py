"""评测指标：多标签 F1 / 精确率 / 召回率 / ECE / Brier / 混淆矩阵。

一个必须修正的口径问题：任务书 §7.4 给的 ECE 公式用 `probs.max(1)` 拿最大类置信度，
那是**单标签**的做法。本任务是**多标签**：一帧可同时命中多类，
按 max(1) 会丢掉其余类的校准信息。

正确做法：把每个 (样本, 类别) 拆成一个二元判定 (p_ij, y_ij)，
逐类分箱统计置信度与实际正确率的加权偏差，再 macro 平均。
这里同时给出两种口径，主表用 multi-label ECE，单标签口径仅作对照。
"""
from __future__ import annotations

import sys
import pathlib
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import CLASSES, NUM_CLASSES  # noqa: E402


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def multihot_from_logits(logits: np.ndarray, thresholds: float | np.ndarray = 0.5) -> np.ndarray:
    probs = sigmoid(logits)
    thr = np.asarray(thresholds)
    if thr.ndim == 0:
        return (probs > thr).astype(np.int8)
    return (probs > thr[None, :]).astype(np.int8)


def macro_f1_from_logits(logits: np.ndarray, targets: np.ndarray, thresholds: float | np.ndarray = 0.5) -> float:
    preds = multihot_from_logits(logits, thresholds)
    return float(macro_f1(preds, np.asarray(targets)))


def macro_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    _, _, f1s = prf_per_class(preds, targets)
    return float(np.mean([f for f in f1s if f is not None]))


def prf_per_class(preds: np.ndarray, targets: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """逐类精确率 / 召回率 / F1。无正例且无预测的类返回 None。"""
    P: list[float] = []
    R: list[float] = []
    F: list[float] = []
    for j in range(preds.shape[1]):
        p = preds[:, j]
        t = targets[:, j]
        tp = float(((p == 1) & (t == 1)).sum())
        fp = float(((p == 1) & (t == 0)).sum())
        fn = float(((p == 0) & (t == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if (tp + fp + fn) > 0 else None)
        P.append(precision)
        R.append(recall)
        F.append(f1)
    return P, R, F


def confusion_stats(preds: np.ndarray, targets: np.ndarray) -> list[dict[str, int]]:
    out = []
    for j in range(preds.shape[1]):
        p, t = preds[:, j], targets[:, j]
        out.append({
            "tp": int(((p == 1) & (t == 1)).sum()),
            "fp": int(((p == 1) & (t == 0)).sum()),
            "fn": int(((p == 0) & (t == 1)).sum()),
            "tn": int(((p == 0) & (t == 0)).sum()),
        })
    return out


# ---------------------------------------------------------------- 校准


def _bin_stats(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            out.append({"lo": float(lo), "hi": float(hi), "count": 0, "conf": None, "acc": None})
        else:
            out.append({
                "lo": float(lo), "hi": float(hi), "count": int(m.sum()),
                "conf": float(conf[m].mean()), "acc": float(correct[m].mean()),
            })
    return out


def ece_multilabel(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    """multi-label ECE：把 (样本, 类别) 全部摊平成二元判定后分箱。"""
    p = probs.reshape(-1)
    y = np.asarray(targets).reshape(-1).astype(np.float32)
    conf, pred = p, (p >= 0.5).astype(np.float32)
    correct = (pred == y).astype(np.float32)
    bins = _bin_stats(conf, correct, n_bins)
    total = len(p)
    e = sum(b["count"] / total * abs(b["acc"] - b["conf"]) for b in bins if b["count"] > 0)
    return float(e), bins


def ece_singlelabel(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """单标签口径的 ECE（按 max 置信度），仅作对照，不进主表。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == np.asarray(targets).argmax(axis=1)).astype(np.float32)
    bins = _bin_stats(conf, correct, n_bins)
    total = len(conf)
    return float(sum(b["count"] / total * abs(b["acc"] - b["conf"]) for b in bins if b["count"] > 0))


def brier_multilabel(probs: np.ndarray, targets: np.ndarray) -> float:
    """逐类 Brier 分数的 macro 平均。"""
    p = sigmoid(np.asarray(probs)) if probs.min() < 0 else np.asarray(probs)
    y = np.asarray(targets).astype(np.float32)
    per_class = ((p - y) ** 2).mean(axis=0)
    return float(per_class.mean())


def reliability_data(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    p = np.asarray(probs).reshape(-1)
    y = np.asarray(targets).reshape(-1).astype(np.float32)
    pred = (p >= 0.5).astype(np.float32)
    return _bin_stats(p, (pred == y).astype(np.float32), n_bins)


# ---------------------------------------------------------------- 阈值搜索


def search_thresholds(
    probs: np.ndarray, targets: np.ndarray, recall_floor: float = 0.95, n_grid: int = 81
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """逐类阈值搜索：在召回率 >= recall_floor 的约束下最大化精确率。

    质检场景里漏检（坏数据流出）的代价远高于误检（多一次人工复核），
    所以阈值不是拍 0.5，而是"召回约束下最大化精确率"。
    """
    p = sigmoid(np.asarray(probs)) if np.asarray(probs).min() < 0 else np.asarray(probs)
    y = np.asarray(targets)
    grid = np.linspace(0.05, 0.95, n_grid)
    thr_out = []
    detail = []
    for j, c in enumerate(CLASSES):
        pj, yj = p[:, j], y[:, j]
        best_thr, best_prec, best_rec = 0.5, -1.0, 0.0
        curve = []
        for thr in grid:
            pred = (pj >= thr).astype(np.float32)
            tp = float(((pred == 1) & (yj == 1)).sum())
            fp = float(((pred == 1) & (yj == 0)).sum())
            fn = float(((pred == 0) & (yj == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            curve.append({"thr": round(float(thr), 3), "precision": round(prec, 4), "recall": round(rec, 4)})
            if rec >= recall_floor and prec > best_prec:
                best_thr, best_prec, best_rec = float(thr), prec, rec
        # 若没有任何阈值满足召回约束（召回本身达不到 floor），退化为取召回最高的阈值
        if best_prec < 0:
            recs = [q["recall"] for q in curve]
            k = int(np.argmax(recs))
            best_thr, best_prec, best_rec = float(grid[k]), curve[k]["precision"], curve[k]["recall"]
            star = False
        else:
            star = True
        thr_out.append(best_thr)
        detail.append({
            "class": c, "threshold": round(best_thr, 3),
            "precision": round(best_prec, 4), "recall": round(best_rec, 4),
            "meets_recall_floor": star, "curve": curve,
        })
    return np.array(thr_out), detail


# ---------------------------------------------------------------- 总入口


def evaluate_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    probs = sigmoid(np.asarray(logits))
    preds = multihot_from_logits(logits, thresholds)
    P, R, F = prf_per_class(preds, np.asarray(targets))
    ece, bins = ece_multilabel(probs, np.asarray(targets))

    per_class = {}
    for j, c in enumerate(CLASSES):
        per_class[c] = {
            "precision": None if P[j] is None else round(P[j], 4),
            "recall": None if R[j] is None else round(R[j], 4),
            "f1": None if F[j] is None else round(F[j], 4),
        }

    return {
        "macro_f1": round(macro_f1(preds, np.asarray(targets)), 4),
        "per_class": per_class,
        "ece_multilabel": round(ece, 4),
        "ece_singlelabel_ref": round(ece_singlelabel(probs, np.asarray(targets)), 4),
        "brier": round(brier_multilabel(probs, np.asarray(targets)), 4),
        "reliability_bins": bins,
        "confusion": confusion_stats(preds, np.asarray(targets)),
        "n": int(len(targets)),
        "thresholds": (None if np.isscalar(thresholds) else [round(float(t), 3) for t in np.asarray(thresholds)]),
    }


def slice_metrics(
    records: list[dict[str, Any]],
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    keys: tuple[str, ...] = ("scene", "difficulty", "source"),
) -> dict[str, Any]:
    """分切片报告：按场景 / 难度 / 来源 切开看指标，暴露均值掩盖掉的问题。"""
    preds = multihot_from_logits(probs, thresholds)
    out: dict[str, Any] = {}
    for key in keys:
        buckets: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            buckets.setdefault(str(r.get(key, "unknown")), []).append(i)
        res = {}
        for b, idx in sorted(buckets.items()):
            sub_pred = preds[idx]
            sub_t = targets[idx]
            P, R, F = prf_per_class(sub_pred, sub_t)
            f1s = [f for f in F if f is not None]
            res[b] = {
                "n": len(idx),
                "macro_f1": round(float(np.mean(f1s)), 4) if f1s else None,
                "macro_recall": round(
                    float(np.mean([r for r in R if r is not None])), 4
                ) if any(r is not None for r in R) else None,
            }
        out[key] = res
    return out
