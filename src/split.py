"""按原图分组切分，杜绝数据泄漏。

核心规则：**同一 origin_id 的所有变体必须落在同一 split**。
一条真实帧会派生出多条合成样本，若按样本随机切分，同一张图的"清晰版"
和"模糊版"会分别落进 train 和 test —— 模型直接背下了答案，
指标虚高且完全不可信。这是本项目最容易踩也最该被面试官追问的坑。

切分顺序（两阶段）：
    阶段一：优先预留 eval 池（按组），目标 ~1400 张，冻结为 eval_v1
    阶段二：剩余组按 7:2:1 切成 train / val / test

这样 eval_v1 与训练完全隔离，同时保留任务书要求的 7:2:1 口径。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    MANIFEST_PATH,
    REPORTS_DIR,
    load_manifest,
    write_json,
)


def group_by_origin(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    g: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        g[r["origin_id"]].append(i)
    return g


def assign_splits(
    records: list[dict[str, Any]],
    eval_target: int = 1400,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """返回 (带 split 字段的记录, 切分报告)。"""
    groups = group_by_origin(records)
    origin_ids = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(origin_ids)

    # ---- 阶段一：预留 eval 池
    eval_origins: list[str] = []
    eval_count = 0
    pool_origins: list[str] = []
    for oid in origin_ids:
        if eval_count < eval_target:
            eval_origins.append(oid)
            eval_count += len(groups[oid])
        else:
            pool_origins.append(oid)

    # ---- 阶段二：剩余按 7:2:1
    n = len(pool_origins)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_origins = set(pool_origins[:n_train])
    val_origins = set(pool_origins[n_train : n_train + n_val])
    test_origins = set(pool_origins[n_train + n_val :])

    def which(oid: str) -> str:
        if oid in train_origins:
            return "train"
        if oid in val_origins:
            return "val"
        if oid in test_origins:
            return "test"
        return "eval"

    out = []
    for r in records:
        r = dict(r)
        r["split"] = which(r["origin_id"])
        out.append(r)

    def _stat(origins) -> dict[str, int]:
        idxs = [i for o in origins for i in groups[o]]
        return {
            "origins": len(origins),
            "samples": len(idxs),
            "real": sum(1 for i in idxs if records[i]["source"] == "real"),
            "procedural": sum(1 for i in idxs if records[i]["source"] == "procedural"),
            "aigc": sum(1 for i in idxs if records[i]["source"] == "aigc"),
        }

    report = {
        "seed": seed,
        "eval_target": eval_target,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
        "eval": _stat(eval_origins),
        "train": _stat(train_origins),
        "val": _stat(val_origins),
        "test": _stat(test_origins),
    }
    return out, report


def check_leakage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """泄漏自检：任一 origin_id 不得跨越多个 split。"""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in records:
        seen[r["origin_id"]].add(r["split"])
    leaked = {o: sorted(s) for o, s in seen.items() if len(s) > 1}
    return {
        "passed": len(leaked) == 0,
        "total_origins": len(seen),
        "leaked_origins": len(leaked),
        "examples": dict(list(leaked.items())[:10]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按原图分组切分")
    ap.add_argument("--eval-target", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    records = load_manifest()
    active = [r for r in records if not r.get("dedup_removed", False)]
    if not active:
        print("[split] manifest 为空")
        return 1

    out, report = assign_splits(active, args.eval_target, seed=args.seed)

    # 去重剔除的样本保留在 manifest 里，split 记为 excluded，便于追溯
    removed = [dict(r, split="excluded") for r in records if r.get("dedup_removed", False)]
    all_records = out + removed

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    leak = check_leakage(out)
    report["leakage_check"] = leak
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(report, REPORTS_DIR / "split_report.json")

    print(f"[split] seed={args.seed}")
    for k in ("eval", "train", "val", "test"):
        s = report[k]
        print(
            f"        {k:<6} 组 {s['origins']:>5}  样本 {s['samples']:>5}  "
            f"(real {s['real']} / proc {s['procedural']} / aigc {s['aigc']})"
        )
    print(f"[split] 泄漏自检: {'通过' if leak['passed'] else '失败'} "
          f"（{leak['total_origins']} 个原图组，跨 split 的 {leak['leaked_origins']} 个）")
    print("[split] 报告 -> reports/split_report.json")
    return 0 if leak["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
