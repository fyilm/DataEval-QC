"""冻结评测集构造：分层抽样 + 长尾补录。

分层网格：缺陷类型(6 + normal) × 场景(室内/室外/夜视) × 难度(易/中/难)

配额分配用"等额 + 按需回补"：先给每个格子等额目标，
取不满的格子（典型是夜视格，夜景原图本来就少）把缺额按比例让给有余量的格子，
保证总数恰好等于目标。这样既有分层口径，又不会因为某一层样本天然稀少而凑不满。

长尾 200 张：逆光 / 雨雾 / 污渍 / 拖影 / 极暗+强压缩 等边缘场景。
本仓库以程序化方式补录并标记 eval_longtail=true，状态为"待人工复核"；
人工复核入口见 src.annotate_tool。

评测集一旦冻结，任何变更必须写进 docs/evalset_changelog.md。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
from collections import defaultdict
from datetime import date
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    CLASSES,
    EVAL_DIR,
    ROOT,
    ensure_dirs,
    rel,
    sha1_of_bytes,
)
from src.degrade import save_derived  # noqa: E402
from src.common import imread, imwrite  # noqa: E402

EVAL_JSONL = EVAL_DIR / "eval_v1.jsonl"   # 规范格式，流水线读这个
EVAL_CSV = EVAL_DIR / "eval_v1.csv"       # 交付物，按需导出（见下方 DLP 说明）
NORMAL = "normal"


# ---------------------------------------------------------------- 分层抽样


def cell_of(rec: dict[str, Any]) -> tuple[str, str, str]:
    """样本所属分层格子：缺陷类型 × 场景 × 难度。

    多标签样本按 params._order 的首类归格，保证每个样本只占一个格子，
    避免同一条样本在多个格子里被重复计数。
    """
    labels = rec.get("labels") or []
    if not labels:
        defect = NORMAL
    else:
        order = (rec.get("params") or {}).get("_order") or labels
        defect = order[0] if order[0] in CLASSES else labels[0]
    scene = rec.get("scene") or "indoor"
    difficulty = rec.get("difficulty") or "mid"
    if defect == NORMAL:
        difficulty = "none"
    return (defect, scene, difficulty)


def allocate_quotas(avail: dict[tuple[str, str, str], int], target: int) -> dict[tuple[str, str, str], int]:
    """等额起步 + 缺额回补，总数严格等于 target（受各格可用量上限约束）。"""
    quota = {c: 0 for c in avail}
    remaining = target
    while remaining > 0:
        eligible = [c for c in avail if quota[c] < avail[c]]
        if not eligible:
            break
        add = max(1, remaining // len(eligible))
        for c in eligible:
            if remaining <= 0:
                break
            take = min(add, avail[c] - quota[c], remaining)
            if take > 0:
                quota[c] += take
                remaining -= take
    return quota


def stratified_sample(
    pool: list[dict[str, Any]], target: int = 1200, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in pool:
        by_cell[cell_of(r)].append(r)

    avail = {c: len(v) for c, v in by_cell.items()}
    quota = allocate_quotas(avail, target)

    picked: list[dict[str, Any]] = []
    cell_stat = {}
    for cell, q in sorted(quota.items()):
        items = by_cell[cell][:]
        rng.shuffle(items)
        take = items[:q]
        picked.extend(take)
        cell_stat["|".join(cell)] = {"available": avail[cell], "quota": q, "taken": len(take)}

    rng.shuffle(picked)
    report = {
        "target": target,
        "achieved": len(picked),
        "cells": len(avail),
        "cell_stat": cell_stat,
        "underfilled_cells": {k: v for k, v in cell_stat.items() if v["taken"] < v["quota"]},
    }
    return picked, report


# ---------------------------------------------------------------- 长尾补录


def _backlit(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """逆光：背景强烈过曝 + 主体压暗成剪影。"""
    h, w = img.shape[:2]
    cy, cx = h * rng.uniform(0.35, 0.65), w * rng.uniform(0.3, 0.7)
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    d = np.sqrt((y - cy / h) ** 2 + (x - cx / w) ** 2)
    glow = np.clip(1.35 - d * 1.6, 0, 1)[:, :, None].astype(np.float32)
    out = img.astype(np.float32) * (1.0 + glow * 1.6)
    silhouette = np.clip(1.0 - np.clip(0.55 - d, 0, 1) * 1.8, 0.18, 1.0)[:, :, None]
    out = out * silhouette
    return np.clip(out, 0, 255).astype(np.uint8)


def _smear(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """强运动拖影。"""
    k = rng.randint(35, 55)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def _harsch_combo(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """极暗 + 强压缩复合：监控夜间低码率典型形态。"""
    f = (img.astype(np.float32) / 255.0) ** rng.uniform(2.4, 3.4)
    out = np.clip(f * 255, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(6, 14))])
    if not ok:
        return out
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


LONGTAIL_KINDS = {
    "backlit": (_backlit, ["over_expo", "dark"]),
    "rain_fog": (None, ["occlusion"]),   # 复用 aigc 的雾/雨
    "smudge": (None, ["occlusion"]),
    "smear": (_smear, ["blur"]),
    "dark_compressed": (_harsch_combo, ["dark", "noise"]),
}


def build_longtail(
    real_pool: list[dict[str, Any]], num: int = 200, seed: int = 42
) -> list[dict[str, Any]]:
    """生成长尾边缘样本。标记为待人工复核。"""
    from src.aigc import _fog, _mud_splatter, _rain

    rng = random.Random(seed)
    out_dir = EVAL_DIR / "longtail"
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = list(LONGTAIL_KINDS.keys())
    records = []

    for i in range(num):
        src = real_pool[i % len(real_pool)]
        img = imread(ROOT / src["file"])
        if img is None:
            continue
        kind = kinds[i % len(kinds)]
        fn, labels = LONGTAIL_KINDS[kind]

        if kind == "rain_fog":
            st = rng.choice(["moderate", "severe"])
            img2 = _rain(img, rng, st) if rng.random() < 0.5 else _fog(img, rng, st)
        elif kind == "smudge":
            img2 = _mud_splatter(img, rng, rng.choice(["moderate", "severe"]))
        else:
            img2 = fn(img, rng)

        params = {
            "_difficulty": "hard",
            "_order": labels,
            "longtail_kind": kind,
        }
        rec = save_derived(
            img2, out_dir, f"lt_{i:05d}", "procedural", src["origin_id"],
            src.get("scene", "indoor"), labels, params, parent_sha1=src["sha1"],
        )
        rec["split"] = "eval"
        rec["eval_longtail"] = True
        rec["review_status"] = "pending_human"
        records.append(rec)
    return records


# ---------------------------------------------------------------- 落盘


def write_eval_jsonl(records: list[dict[str, Any]], path: pathlib.Path = EVAL_JSONL) -> pathlib.Path:
    """写规范格式 JSONL。

    重要：这台机器装有企业 DLP 透明加密，.csv/.xls/.doc 等 doc 类型文件会在
    落地几分钟后被加密，且我们的进程读回时拿到的是密文（%TSD-Header-###% 头），
    直接 UnicodeDecodeError。.jsonl/.json/.jpg/.py/.md/.yaml 不受影响。
    所以机器可读的规范格式必须用 .jsonl，CSV 只作为给人看的导出物。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    dropped = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            # 兜底去重：同一张图在评测集里出现两次会让它的权重翻倍，
            # 指标失真。抽样端已排除长尾，这里再兜一道。
            if r["file"] in seen:
                dropped += 1
                continue
            seen.add(r["file"])
            written += 1
            defect, scene, difficulty = cell_of(r)
            f.write(
                json.dumps(
                    {
                        "file": r["file"], "sha1": r["sha1"], "source": r["source"],
                        "origin_id": r["origin_id"], "scene": scene, "difficulty": difficulty,
                        "defect_type": defect, "labels": r.get("labels") or [],
                        "longtail": bool(r.get("eval_longtail")),
                        "review_status": r.get("review_status", "auto"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if dropped:
        print(f"[evalset] 写入 {written} 条，去重丢弃 {dropped} 条重复样本")
    return path


def write_eval_csv(records: list[dict[str, Any]], path: pathlib.Path = EVAL_CSV) -> pathlib.Path:
    """导出人类可读 CSV。注意本机 DLP 会在几分钟后加密 .csv，推开源前请检查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "file", "sha1", "source", "origin_id", "scene", "difficulty",
        "defect_type", "labels", "longtail", "review_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            defect, scene, difficulty = cell_of(r)
            w.writerow([
                r["file"], r["sha1"], r["source"], r["origin_id"], scene, difficulty,
                defect, "|".join(r.get("labels") or []),
                "1" if r.get("eval_longtail") else "0",
                r.get("review_status", "auto"),
            ])
    return path


def append_changelog(note: str, stats: dict[str, Any]) -> None:
    p = ROOT / "docs" / "evalset_changelog.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"\n## eval_v1 — {date.today().isoformat()}\n\n"
        f"{note}\n\n"
        f"- 样本数：{stats.get('achieved', 0)}\n"
        f"- 分层格数：{stats.get('cells', 0)}\n"
        f"- 抽样 seed：{stats.get('seed', 42)}\n"
    )
    if not p.exists():
        p.write_text("# 评测集版本变更记录\n\n> 评测集冻结后，任何变更必须在此登记。\n", encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(entry)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构造并冻结评测集")
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--longtail", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.common import load_manifest

    ensure_dirs()
    records = load_manifest()
    if not records:
        print("[evalset] manifest 为空")
        return 1

    # 必须排除长尾样本：本脚本可重复运行，而长尾样本在首次运行后已被写进
    # manifest（split=eval）。若不排除，第二次运行时它们会进入抽样池，被分层
    # 抽样抽中一部分，随后 build_longtail 又按固定文件名生成一遍同名样本，
    # 导致同一张图在评测集里出现两次 —— 实测 1400 行里只有 1277 个唯一文件，
    # 123 张被重复计权。
    pool = [
        r for r in records
        if r.get("split") == "eval" and not r.get("dedup_removed")
        and not r.get("eval_longtail")
    ]
    if not pool:
        print("[evalset] eval 池为空，请先运行 split")
        return 1

    picked, report = stratified_sample(pool, args.target, args.seed)
    for r in picked:
        r["review_status"] = "auto"

    lt = []
    if args.longtail > 0:
        # 长尾必须来自与 eval 池相同的原图集合，否则会引入新的原图组造成泄漏风险
        eval_origins = {r["origin_id"] for r in pool}
        real_pool = [
            r for r in records
            if r["source"] == "real" and r["origin_id"] in eval_origins
            and not r.get("dedup_removed")
        ]
        if not real_pool:
            real_pool = [r for r in records if r["source"] == "real"]
        lt = build_longtail(real_pool, args.longtail, args.seed)
        for r in lt:
            r["defect_override"] = None

    all_eval = picked + lt
    write_eval_jsonl(all_eval)
    write_eval_csv(all_eval)

    # 把长尾样本补进 manifest，并置为 eval
    import json
    from src.common import MANIFEST_PATH

    known = {r["file"] for r in records}
    new_recs = [r for r in lt if r["file"] not in known]
    if new_recs:
        with MANIFEST_PATH.open("a", encoding="utf-8") as f:
            for r in new_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report["seed"] = args.seed
    report["longtail"] = len(lt)
    report["total"] = len(all_eval)
    from src.common import write_json

    write_json(report, ROOT / "reports" / "evalset_report.json")
    append_changelog(
        f"首次冻结。分层抽样 {len(picked)} 张 + 长尾补录 {len(lt)} 张，共 {len(all_eval)} 张。",
        report,
    )

    print(
        f"[evalset] 分层抽样 {len(picked)}/{args.target}，长尾 {len(lt)}，合计 {len(all_eval)}\n"
        f"          分层格数 {report['cells']}，未填满的格子 {len(report['underfilled_cells'])} 个\n"
        f"          -> {EVAL_JSONL.relative_to(ROOT)}（规范格式，流水线读这个）\n"
        f"          -> {EVAL_CSV.relative_to(ROOT)}（导出用，本机 DLP 可能加密 .csv）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
