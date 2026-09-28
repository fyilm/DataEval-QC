"""校验换机器后重新生成的数据集与仓库里的一致。

背景：数据集图片（约 810MB）不入库，换机器后要么重新生成、要么拷贝 data/。
但重新生成的数据如果和原来不一样，冻结的评测集就失去意义 —— 此前的指标
不再可比。所以必须能验证。

基准来自 git（已入库的 manifest.jsonl / eval_v1.jsonl），不需要额外维护
baseline 文件：git 本身就是基准。

用法：
    uv run python scripts/verify_reproduce.py                 # 两个都校验
    uv run python scripts/verify_reproduce.py --manifest-only
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACKED = {
    "manifest": "data/manifest.jsonl",
    "eval": "data/eval/eval_v1.jsonl",
}


def git_show(path: str) -> str | None:
    """取 HEAD 里的文件内容；未入库或不在 git 仓库中返回 None。"""
    try:
        r = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        )
    except OSError:
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def load_index(text: str, key: str = "sha1") -> dict[str, str]:
    """把 jsonl 解析成 {file: sha1}。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        f = d.get("file")
        if f:
            out[f] = str(d.get(key, ""))
    return out


def compare(name: str, rel: str) -> dict[str, Any]:
    base_text = git_show(rel)
    if base_text is None:
        return {"name": name, "ok": False, "reason": f"git 里没有 {rel}（未入库或不在 git 仓库）"}

    cur_path = ROOT / rel
    if not cur_path.exists():
        return {"name": name, "ok": False, "reason": f"本地缺少 {rel}，请先跑数据管线"}

    base = load_index(base_text)
    cur = load_index(cur_path.read_text(encoding="utf-8"))

    missing = sorted(set(base) - set(cur))
    extra = sorted(set(cur) - set(base))
    common = set(base) & set(cur)
    mismatched = sorted(f for f in common if base[f] and cur[f] and base[f] != cur[f])

    ok = not missing and not mismatched
    return {
        "name": name,
        "ok": ok,
        "baseline_n": len(base),
        "current_n": len(cur),
        "matched": len(common) - len(mismatched),
        "missing": missing[:10],
        "missing_n": len(missing),
        "extra_n": len(extra),
        "mismatched": mismatched[:10],
        "mismatched_n": len(mismatched),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="校验重新生成的数据集与仓库基准是否一致")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    targets = ["manifest"] if args.manifest_only else list(TRACKED)
    results = [compare(n, TRACKED[n]) for n in targets]

    all_ok = True
    for r in results:
        print(f"\n=== {r['name']} ({TRACKED[r['name']]}) ===")
        if "reason" in r:
            print(f"  ⚠ {r['reason']}")
            all_ok = False
            continue
        status = "✅ 一致" if r["ok"] else "❌ 不一致"
        print(f"  {status}：基准 {r['baseline_n']} 条 / 当前 {r['current_n']} 条，"
              f"内容匹配 {r['matched']} 条")
        if r["missing_n"]:
            print(f"  缺失 {r['missing_n']} 个（前 10）：{r['missing']}")
        if r["extra_n"]:
            print(f"  多出 {r['extra_n']} 个（重新生成时样本数可能变化）")
        if r["mismatched_n"]:
            print(f"  内容不一致 {r['mismatched_n']} 个（前 10）：{r['mismatched']}")
        if not r["ok"]:
            all_ok = False

    print()
    if all_ok:
        print("数据集与仓库基准一致，此前指标可直接对比。")
        return 0
    print("数据集与基准不一致 —— 换机器后重新生成出现了偏差。")
    print("常见原因：COCO parquet 分片内容变动、采集时网络中断导致样本数不同、")
    print("或数据管线 seed 被改动。若评测集清单变了，请按 docs/evalset_changelog.md 登记新版本。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
