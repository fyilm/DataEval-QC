"""pHash 跨源去重。

注意一个必须绕开的坑：程序化/编辑变异样本都是从真实帧变出来的，
和母图天然高度相似。所以全局按 pHash 去重会把合成数据几乎删光。

正确口径：**只在 origin_id 不同的样本对之间比对**。
这样留下来的重复项才是真正的重复（同一张真实图被采集了两次、
或不同原图意外产生了近乎一致的帧），而同一原图的合法变体全部保留。

实现：把 64 位 pHash 摊成 uint64 数组，用 numpy 分块算两两汉明距离，
精确且比逐对 Python 循环快两个数量级。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np
import PIL.Image
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    MANIFEST_PATH,
    REPORTS_DIR,
    ROOT,
    load_manifest,
    write_json,
)

# 来源优先级：重复组里保留优先级高的那个
SOURCE_PRIORITY = {"real": 0, "procedural": 1, "aigc": 2}


def compute_phashes(records: list[dict[str, Any]], hash_size: int = 8) -> np.ndarray:
    """逐张算 pHash，返回 uint64 数组。"""
    import imagehash

    out = np.zeros(len(records), dtype=np.uint64)
    for i, rec in enumerate(tqdm(records, desc="计算 pHash")):
        p = ROOT / rec["file"]
        try:
            with PIL.Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
            out[i] = int(str(h), 16)
        except Exception:
            out[i] = 0
    return out


def pairwise_hamming_le(
    hashes: np.ndarray, threshold: int, chunk: int = 512
) -> list[tuple[int, int, int]]:
    """找出所有汉明距离 <= threshold 的样本对。分块避免一次性撑爆内存。"""
    n = len(hashes)
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sub = hashes[start:end]
        xor = np.bitwise_xor(sub[:, None], hashes[None, :])
        dist = np.bitwise_count(xor).astype(np.int16)
        ii, jj = np.nonzero(dist <= threshold)
        for a, b in zip(ii.tolist(), jj.tolist()):
            ga, gb = start + a, b
            if ga < gb:  # 每对只记一次，且跳过自身
                pairs.append((ga, gb, int(dist[a, b])))
    return pairs


def dedup(
    records: list[dict[str, Any]],
    hashes: np.ndarray,
    threshold: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行去重，返回 (去重后的记录列表, 去重报告)。"""
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_sha1: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_sha1.setdefault(r.get("sha1", ""), []).append(i)

    dup_pairs = pairwise_hamming_le(hashes, threshold)

    same_origin_skipped = 0
    cross_origin_pairs: list[tuple[int, int, int]] = []
    for i, j, d in dup_pairs:
        if records[i]["origin_id"] == records[j]["origin_id"]:
            same_origin_skipped += 1  # 同一原图的合法变体，不去重
            continue
        union(i, j)
        cross_origin_pairs.append((i, j, d))

    # 同一文件哈希完全相同的一定是重复
    exact_groups = {k: v for k, v in by_sha1.items() if len(v) > 1 and k}

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    removed: set[int] = set()
    removed_detail: list[dict[str, Any]] = []
    for root_idx, members in groups.items():
        if len(members) <= 1:
            continue
        # 保留：来源优先级高者 > 文件序号小者
        keeper = min(
            members,
            key=lambda i: (SOURCE_PRIORITY.get(records[i]["source"], 9), records[i]["file"]),
        )
        for i in members:
            if i == keeper:
                continue
            removed.add(i)
            removed_detail.append(
                {
                    "removed_file": records[i]["file"],
                    "kept_file": records[keeper]["file"],
                    "removed_origin": records[i]["origin_id"],
                    "kept_origin": records[keeper]["origin_id"],
                    "hamming": int(np.bitwise_count(hashes[i] ^ hashes[keeper])),
                }
            )

    kept = []
    for i, r in enumerate(records):
        r = dict(r)
        r["phash"] = f"{int(hashes[i]):016x}"
        r["dedup_removed"] = i in removed
        if i in removed:
            r["dedup_reason"] = f"hamming<={threshold} 且 origin_id 不同"
        kept.append(r)

    report = {
        "threshold": threshold,
        "total": n,
        "removed": len(removed),
        "kept": n - len(removed),
        "same_origin_pairs_skipped": same_origin_skipped,
        "cross_origin_dup_pairs": len(cross_origin_pairs),
        "exact_sha1_dup_groups": len(exact_groups),
        "removed_by_source": _count_by(removed_detail, records),
        "removed_detail_head": removed_detail[:50],
    }
    return kept, report


def _count_by(detail: list[dict], records: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    file_to_src = {r["file"]: r["source"] for r in records}
    for d in detail:
        s = file_to_src.get(d["removed_file"], "?")
        out[s] = out.get(s, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pHash 跨源去重")
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--hash-size", type=int, default=8)
    args = ap.parse_args(argv)

    records = load_manifest()
    if not records:
        print("[dedup] manifest 为空")
        return 1

    hashes = compute_phashes(records, args.hash_size)
    kept, report = dedup(records, hashes, args.threshold)

    import json

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(report, REPORTS_DIR / "dedup_report.json")

    print(
        f"[dedup] 共 {report['total']} 条，剔除 {report['removed']} 条，剩余 {report['kept']} 条\n"
        f"        跨源近似对 {report['cross_origin_dup_pairs']} 对；"
        f"同原图变体 {report['same_origin_pairs_skipped']} 对（按设计保留，不去重）\n"
        f"        按来源剔除: {report['removed_by_source']}\n"
        f"        报告 -> reports/dedup_report.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
