"""DataEval-QC 命令行入口。

各步骤也可以单独调用（uv run python -m src.<module>），这里只是把它们收拢到
一个入口，方便查看和串联。

用法：
    uv run python main.py list        # 列出全部步骤
    uv run python main.py data        # 只跑数据管线
    uv run python main.py all         # 数据 + 实验（等同 scripts/run_all.sh）
"""

from __future__ import annotations

import subprocess
import sys

# (步骤名, 命令参数) —— 与 scripts/run_all.sh 保持一致
DATA_STEPS: list[tuple[str, list[str]]] = [
    ("采集真实清晰帧", ["-m", "src.collect", "--num", "1200", "--night-quota", "300"]),
    ("程序化降级合成", ["-m", "src.degrade", "--num", "6000"]),
    ("第三源：编辑变异", ["-m", "src.aigc", "--num", "2000", "--backend", "edit"]),
    ("pHash 跨源去重", ["-m", "src.dedup"]),
    ("分组切分 + 泄漏自检", ["-m", "src.split"]),
    ("分层抽样冻结评测集", ["-m", "src.evalset"]),
]

EXP_STEPS: list[tuple[str, list[str]]] = [
    ("吞吐基准", ["-m", "src.run_exp", "--mode", "benchmark"]),
    ("全量消融 30 组", ["-m", "src.run_exp", "--config", "configs/ablation.yaml", "--mode", "grid"]),
    ("置信度校准", ["-m", "src.calibrate", "--tag", "best"]),
    ("一致性核对 rule", ["-m", "src.label_vlm", "--backend", "rule", "--limit", "400",
                         "--calibrate", "--out", "vlm_consistency_rule.json"]),
    ("一致性核对 clip", ["-m", "src.label_vlm", "--backend", "clip", "--limit", "400",
                         "--out", "vlm_consistency_clip.json"]),
]


def _run(name: str, args: list[str]) -> bool:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}", flush=True)
    r = subprocess.run([sys.executable, *args])
    if r.returncode != 0:
        print(f"[main] 步骤失败: {name}（退出码 {r.returncode}）", file=sys.stderr)
        return False
    return True


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        print("数据管线：")
        for i, (n, _) in enumerate(DATA_STEPS, 1):
            print(f"  {i}. {n}")
        print("实验：")
        for i, (n, _) in enumerate(EXP_STEPS, 1):
            print(f"  {len(DATA_STEPS) + i}. {n}")
        print("\n运行：uv run python main.py data | all")
        return 0

    if cmd == "data":
        steps = DATA_STEPS
    elif cmd == "all":
        steps = DATA_STEPS + EXP_STEPS
    else:
        print(f"未知命令: {cmd}（可选 list / data / all）")
        return 1

    for name, args in steps:
        if not _run(name, args):
            return 1
    print("\n全部完成。报告在 reports/。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
