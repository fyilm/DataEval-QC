"""标注一致性度量：多标签 Cohen's Kappa。

多标签场景不能直接套用单标签 Kappa。正确做法是把每个 (样本, 类别) 拆成
一个二元判定，逐类算二元 Kappa，再 macro 平均。

完整闭环分三步：
    1) export —— 抽样并导出两份**独立**标注表（A/B），各自只有 file,labels
    2) 两位标注员分别填写（互不可见，保证独立）
    3) merge + kappa —— 合并成 file,annotator1,annotator2 再算一致性

为什么导出两份而不是一份填两列：填在同一张表里就不是独立标注了，
Kappa 会被人为抬高。

关于文件格式：默认导出 .csv（Excel 友好）。但部分企业环境有 DLP 透明加密，
会按后缀异步加密 .csv（文件头变成 %TSD-Header-###%），导致读回时报
UnicodeDecodeError。因此同时导出一份 .jsonl 备份，并在读取时主动检测加密头。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    CLASSES,
    DATA_DIR,
    REPORTS_DIR,
    labels_to_multihot,
    write_json,
)

ANNOTATION_DIR = DATA_DIR / "annotation"
DLP_HEADER = b"%TSD-Header"


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


def _check_dlp(path: pathlib.Path) -> None:
    """检测文件是否被 DLP 透明加密，早报错而不是丢出一个难懂的解码异常。"""
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except OSError:
        return
    if head.startswith(DLP_HEADER):
        raise RuntimeError(
            f"文件被 DLP 透明加密，无法读取：{path}\n"
            "该企业环境会按后缀异步加密 .csv/.xls/.doc 等文档类型。解决办法：\n"
            "  1. 改用同目录的 .jsonl 备份（不会被加密）；\n"
            "  2. 或把标注结果存成 .txt / .json 后再读回。"
        )


def read_annotations(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """读一份 file,annotator1,annotator2 的表，返回 (multi-hot1, multi-hot2, 文件列表)。"""
    rows, _ = _read_rows(pathlib.Path(path))
    f1: list[list[int]] = []
    f2: list[list[int]] = []
    files: list[str] = []
    for row in rows:
        files.append((row.get("file") or "").strip())

        def parse(key: str) -> list[str]:
            v = row.get(key)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return [x.strip() for x in str(v or "").replace(",", "|").split("|") if x.strip()]

        f1.append(labels_to_multihot(parse("annotator1")))
        f2.append(labels_to_multihot(parse("annotator2")))
    return np.array(f1), np.array(f2), files


def _read_rows(path: pathlib.Path) -> tuple[list[dict[str, str]], list[str]]:
    """按后缀读取标注行，统一返回 (行字典列表, 字段名列表)。

    支持三种后缀：
      .jsonl —— 首选，DLP 不加密，程序读写都安全
      .csv   —— Excel 友好，但本环境会被 DLP 加密（读了会报错）
      .txt   —— 不加密，Excel 另存为「制表符分隔」即可
    """
    p = pathlib.Path(path)
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        fields = list(rows[0].keys()) if rows else ["file", "labels"]
        return rows, fields

    _check_dlp(p)
    # txt 可能是 tab 或逗号分隔，嗅探一下
    delim = ","
    if p.suffix.lower() == ".txt":
        sample = p.open(encoding="utf-8-sig").readline()
        delim = "\t" if "\t" in sample else ","
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def read_single(path: str | pathlib.Path) -> tuple[dict[str, list[str]], list[str]]:
    """读一份 file,labels 的单人标注表，返回 {file: [labels]} 与顺序。"""
    rows, _ = _read_rows(pathlib.Path(path))
    out: dict[str, list[str]] = {}
    order: list[str] = []
    for row in rows:
        fn = (row.get("file") or "").strip()
        if not fn:
            continue
        raw = row.get("labels")
        if isinstance(raw, list):        # jsonl 里直接是数组
            labels = [str(x).strip() for x in raw if str(x).strip()]
        else:
            labels = [x.strip() for x in str(raw or "").replace(",", "|").split("|") if x.strip()]
        out[fn] = labels
        order.append(fn)
    return out, order


def export_task(
    records: list[dict[str, Any]],
    n: int,
    out_dir: pathlib.Path,
    seed: int = 42,
) -> dict[str, Any]:
    """抽样并导出 A/B 两份独立标注表。

    导出表**不含规则标签** —— 给人看真实标签会造成锚定，Kappa 会被抬高。
    真实标签单独存进 reports/annotation_gt.json，只用于后续分析，不给标注员。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n, len(records)), replace=False)
    sample = [records[i] for i in sorted(idx)]

    out_dir.mkdir(parents=True, exist_ok=True)
    files = [r["file"] for r in sample]

    for tag in ("A", "B"):
        # csv 便于 Excel 填写；jsonl 是 DLP 环境下的兜底
        with (out_dir / f"annotator_{tag}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "labels"])
            for fn in files:
                w.writerow([fn, ""])
        with (out_dir / f"annotator_{tag}.jsonl").open("w", encoding="utf-8") as f:
            for fn in files:
                f.write(json.dumps({"file": fn, "labels": []}, ensure_ascii=False) + "\n")

    gt = {r["file"]: (r.get("labels") or []) for r in sample}
    gt_path = REPORTS_DIR / "annotation_gt.json"
    write_json(gt, gt_path)

    return {
        "n_sampled": len(sample),
        "seed": seed,
        "files_csv": [f"annotator_{t}.csv" for t in ("A", "B")],
        "files_jsonl": [f"annotator_{t}.jsonl" for t in ("A", "B")],
        "gt_saved_to": str(gt_path),
        "hint": "多标签用 | 分隔，如 blur|noise；无缺陷留空。",
    }


def merge_annotations(
    a_path: str | pathlib.Path,
    b_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
) -> int:
    """把两份独立标注表合并成 file,annotator1,annotator2。

    输出格式由 out_path 后缀决定；默认用 .jsonl —— 本环境里 .csv 会被
    DLP 加密，合并完自己都读不回来。
    """
    a, order = read_single(a_path)
    b, _ = read_single(b_path)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".jsonl":
        with out.open("w", encoding="utf-8") as f:
            for fn in order:
                f.write(json.dumps(
                    {"file": fn, "annotator1": a.get(fn, []), "annotator2": b.get(fn, [])},
                    ensure_ascii=False,
                ) + "\n")
    else:
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "annotator1", "annotator2"])
            for fn in order:
                w.writerow([fn, "|".join(a.get(fn, [])), "|".join(b.get(fn, []))])
    return len(order)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="多标签 Cohen's Kappa：导出 / 合并 / 计算")
    ap.add_argument("--csv", default=None, help="file,annotator1,annotator2 格式的 CSV")
    ap.add_argument("--export", type=int, default=0, metavar="N",
                    help="从评测集抽样 N 张，导出 A/B 两份独立标注表")
    ap.add_argument("--merge", nargs=2, default=None, metavar=("A", "B"),
                    help="合并两份单人标注表为 file,annotator1,annotator2")
    ap.add_argument("--out", default=str(REPORTS_DIR / "kappa_report.json"))
    ap.add_argument("--merged", default=str(ANNOTATION_DIR / "merged.jsonl"), help="合并结果输出路径")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    # 步骤 1：导出标注任务
    if args.export:
        from src.evalset import EVAL_JSONL

        if not EVAL_JSONL.exists():
            print(f"[kappa] 找不到评测集 {EVAL_JSONL}，请先跑 src.evalset")
            return 1
        records = [json.loads(l) for l in EVAL_JSONL.open(encoding="utf-8") if l.strip()]
        info = export_task(records, args.export, ANNOTATION_DIR, args.seed)
        print(f"[kappa] 已抽样 {info['n_sampled']} 张，导出到 {ANNOTATION_DIR}")
        for t in ("A", "B"):
            print(f"        标注员{t}：annotator_{t}.csv（Excel 填） / annotator_{t}.jsonl（程序读）")
        print(f"        填写说明：{info['hint']}")
        print(f"        真实标签已单独留存 -> {info['gt_saved_to']}（不要给标注员看，会造成锚定）")
        print("        ⚠ 本环境会加密 .csv：在 Excel 里填完后，请另存为 .txt（制表符分隔）"
              "或直接改填 .jsonl，否则程序读回来是密文。")
        return 0

    # 步骤 2：合并
    if args.merge:
        n = merge_annotations(args.merge[0], args.merge[1], args.merged)
        print(f"[kappa] 已合并 {n} 条 -> {args.merged}")
        print(f"        下一步：uv run python -m src.agreement --csv {args.merged}")
        return 0

    # 步骤 3：计算
    if not args.csv:
        print("请指定 --export N（导出任务）或 --csv 文件（计算 Kappa）")
        return 1

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
