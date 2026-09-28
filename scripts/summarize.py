"""汇总消融实验结果，产出主表并校验任务书指标门槛。

用法：
    uv run python scripts/summarize.py
    uv run python scripts/summarize.py --results reports/experiment_results.jsonl

输出：
    reports/ablation_summary.json   结构化汇总
    reports/ablation_table.md       可直接粘进报告的 markdown 主表
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import REPORTS_DIR  # noqa: E402

# 任务书的指标门槛
TARGET_MACRO_F1 = 0.85
TARGET_RECALL = 0.95
# 质检最在意的两类（任务书点名）
KEY_CLASSES = ("blur", "low_res")


def _load(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if "error" not in d:
            rows.append(d)
    return rows


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    a = np.array(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("model", "?"), float(r.get("ratio", 0.0)))].append(r)

    table: list[dict[str, Any]] = []
    for (model, ratio), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        f1s = [i["eval"]["macro_f1"] for i in items]
        eces = [i["eval"]["ece_multilabel"] for i in items]
        briers = [i["eval"].get("brier", 0.0) for i in items]
        recalls: dict[str, list[float]] = defaultdict(list)
        for i in items:
            for c, v in i["eval"]["per_class"].items():
                if v.get("recall") is not None:
                    recalls[c].append(float(v["recall"]))
        rec_mean = {c: _mean_std(v)[0] for c, v in recalls.items()}
        m, s = _mean_std(f1s)
        em, _ = _mean_std(eces)
        bm, _ = _mean_std(briers)
        table.append({
            "model": model,
            "ratio": ratio,
            "n_seeds": len(items),
            "n_train": items[0]["mix"]["n_total"],
            "macro_f1_mean": round(m, 4),
            "macro_f1_std": round(s, 4),
            "ece_mean": round(em, 4),
            "brier_mean": round(bm, 4),
            "recall_mean": {c: round(v, 4) for c, v in rec_mean.items()},
            "train_seconds_mean": round(float(np.mean([i["train_seconds"] for i in items])), 1),
        })

    # 达标检查
    best = max(table, key=lambda r: r["macro_f1_mean"]) if table else None
    checks = []
    if best:
        checks.append({
            "item": "macro-F1 ≥ 0.85",
            "target": TARGET_MACRO_F1,
            "best": best["macro_f1_mean"],
            "passed": best["macro_f1_mean"] >= TARGET_MACRO_F1,
        })
        for c in KEY_CLASSES:
            v = best["recall_mean"].get(c)
            checks.append({
                "item": f"{c} 召回 ≥ 0.95",
                "target": TARGET_RECALL,
                "best": v,
                "passed": bool(v is not None and v >= TARGET_RECALL),
            })

    return {
        "n_runs": len(rows),
        "n_groups": len(table),
        "table": table,
        "best_group": best,
        "acceptance_checks": checks,
    }


def to_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 消融实验主表\n")
    lines.append(f"共 {summary['n_runs']} 次训练，{summary['n_groups']} 个（模型 × 合成占比）组合，"
                 f"macro-F1 与 ECE 为对应 seed 的均值。\n")
    lines.append("| 模型 | 合成占比 r | 训练量 | macro-F1 | ECE | Brier | "
                 + " | ".join(f"{c} 召回" for c in KEY_CLASSES) + " |")
    lines.append("|---|---|---|---|---|---|" + "---|" * len(KEY_CLASSES))
    for r in summary["table"]:
        cells = [f"{r['recall_mean'].get(c, float('nan')):.3f}" for c in KEY_CLASSES]
        lines.append(
            f"| {r['model']} | {r['ratio']:.2f} | {r['n_train']} | "
            f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | "
            f"{r['ece_mean']:.4f} | {r['brier_mean']:.4f} | " + " | ".join(cells) + " |"
        )

    lines.append("\n## 任务书指标门槛校验\n")
    lines.append("| 验收项 | 门槛 | 当前最佳 | 结论 |")
    lines.append("|---|---|---|---|")
    for c in summary["acceptance_checks"]:
        best = "n/a" if c["best"] is None else f"{c['best']:.4f}"
        lines.append(f"| {c['item']} | {c['target']} | {best} | {'✅ 达标' if c['passed'] else '❌ 未达标'} |")

    lines.append("\n> 注：r=0 组训练集无正样本（真实帧全部是干净帧），macro-F1 为 0 属设计内的退化对照点。")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总消融实验结果")
    ap.add_argument("--results", default=str(REPORTS_DIR / "experiment_results.jsonl"))
    args = ap.parse_args()

    rows = _load(pathlib.Path(args.results))
    if not rows:
        print(f"[summarize] 没有可用的实验结果：{args.results}")
        return 1

    summary = summarize(rows)
    out_json = REPORTS_DIR / "ablation_summary.json"
    out_md = REPORTS_DIR / "ablation_table.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(to_markdown(summary), encoding="utf-8")

    print(f"[summarize] {summary['n_runs']} 次训练 -> {summary['n_groups']} 个组合")
    for r in summary["table"]:
        print(f"  {r['model']:<22} r={r['ratio']:.2f}  "
              f"macroF1={r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}  ECE={r['ece_mean']:.4f}")
    print()
    for c in summary["acceptance_checks"]:
        best = "n/a" if c["best"] is None else f"{c['best']:.4f}"
        print(f"  {'✅' if c['passed'] else '❌'} {c['item']:<20} 最佳 {best}")
    print(f"\n[summarize] 主表 -> {out_md}")
    print(f"[summarize] 结构化 -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
