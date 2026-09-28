#!/usr/bin/env bash
# 一键复现：从零重建数据集 -> 切分 -> 冻结评测集 -> 消融实验 -> 校准 -> 一致性核对
#
# 全部步骤在固定 seed 下运行，重复执行结果一致。
# 用法：
#   bash scripts/run_all.sh          # 全跑（CPU 上约 3 小时，主要是消融实验）
#   bash scripts/run_all.sh --data   # 只跑数据管线（约 10 分钟）
#
# 前置：已安装 uv（https://docs.astral.sh/uv/），首次运行会自动建虚拟环境。
set -euo pipefail
cd "$(dirname "$0")/.."

ONLY_DATA=0
if [[ "${1:-}" == "--data" ]]; then
  ONLY_DATA=1
fi

step() {
  echo ""
  echo "=============================================================="
  echo "  $1"
  echo "=============================================================="
}

# ---------------------------------------------------------------- 数据管线
step "1/10 采集真实清晰帧（COCO，含夜间专项配额）"
uv run python -m src.collect --num 1200 --night-quota 300

step "2/10 程序化降级合成"
uv run python -m src.degrade --num 6000

step "3/10 第三源：编辑变异（GPU 机器上可改 --backend sd 走真实扩散）"
uv run python -m src.aigc --num 2000 --backend edit

step "4/10 pHash 跨源去重"
uv run python -m src.dedup

step "5/10 按母图分组切分 + 泄漏自检"
uv run python -m src.split

step "6/10 分层抽样并冻结评测集"
uv run python -m src.evalset

if [[ "$ONLY_DATA" == "1" ]]; then
  echo ""
  echo "数据管线完成。manifest -> data/manifest.jsonl，评测集 -> data/eval/eval_v1.jsonl"
  exit 0
fi

# ---------------------------------------------------------------- 实验
step "7/10 吞吐基准（用于给消融定档）"
uv run python -m src.run_exp --mode benchmark

step "8/10 全量消融（2 模型 x 5 合成占比 x 3 seed = 30 组）"
uv run python -m src.run_exp --config configs/ablation.yaml --mode grid

step "9/10 置信度校准（温度标定 + 可靠性图）"
uv run python -m src.calibrate --tag best

step "10/10 预标注一致性核对（rule 校准版 + CLIP）"
uv run python -m src.label_vlm --backend rule --limit 400 --calibrate --out vlm_consistency_rule.json
uv run python -m src.label_vlm --backend clip  --limit 400 --out vlm_consistency_clip.json

echo ""
echo "全部完成。报告在 reports/，图在 reports/figures/。"
