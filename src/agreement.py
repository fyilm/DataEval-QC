"""标注一致性度量：多标签 Cohen's Kappa。

多标签场景不能直接套用单标签 Kappa。正确做法是把每个 (样本, 类别) 拆成
一个二元判定，逐类算二元 Kappa，再 macro 平均。

输入是两份"标注员对同一批样本的标签"，标注员可以是真人，也可以是
AI 辅助标注 + 人工复核的组合。读入 CSV 格式：
    file,annotator1,annotator2
其中 annotator 列用 | 分隔多标签，如 blur|noise。
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    CLASSES,
    REPORTS_DIR,
    labels_to_multihot,
    write_json,
)


def binary_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """二元 Cohen's Kappa。全同类别（无正例或无负例）时定义为 1.0。"""
    n = len(y1)
    if n == 0:
        return 0.0
    po = float((y1 == y2).mean())
    p1_pos = float(y1.mean())
    p2_pos = float(y2.mean())
    pe = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    if pe >= 1.0 - 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return (po - pe) / (1 - pe)


def multilabel_kappa(y1: np.ndarray, y2: np.ndarray, classes: list[str] | None = None) -> dict[str, Any]:
    """逐类二元 Kappa + macro 平均。

    y1, y2: (n_samples, n_classes) 的 multi-hot 矩阵。
    """
    if classes is None:
        classes = CLASSES
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    per_class: dict[str, float] = {}
    for j, c in enumerate(classes):
        per_class[c] = round(binary_kappa(y1[:, j], y2[:, j]), 4)
    macro = float(np.mean(list(per_class.values())))
    return {"per_class": per_class, "macro_kappa": round(macro, 4)}


def read_annotations(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """读两标注员重叠标注 CSV，返回 (multi-hot1, multi-hot2, 文件名列表)。"""
    f1: list[list[int]] = []
    f2: list[list[int]] = []
    files: list[str] = []
    with pathlib.Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            files.append(row.get("file", ""))
            f1.append(labels_to_multihot([x for x in (row.get("annotator1") or "").split("|") if x]))
            f2.append(labels_to_multihot([x for x in (row.get("annotator2") or "").split("|") if x]))
    return np.array(f1), np.array(f2), files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="计算多标签 Cohen's Kappa")
    ap.add_argument("--csv", required=True, help="file,annotator1,annotator2 格式的 CSV")
    ap.add_argument("--out", default=str(REPORTS_DIR / "kappa_report.json"))
    args = ap.parse_args(argv)

    y1, y2, files = read_annotations(args.csv)
    if len(files) == 0:
        print("[kappa] CSV 里没有样本")
        return 1

    res = multilabel_kappa(y1, y2)
    res["n_samples"] = len(files)
    write_json(res, args.out)

    print(f"[kappa] 样本数 {res['n_samples']}")
    for c, k in res["per_class"].items():
        print(f"          {c:<10} {k:.4f}")
    print(f"          macro Kappa = {res['macro_kappa']:.4f}")
    print(f"[kappa] 报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
