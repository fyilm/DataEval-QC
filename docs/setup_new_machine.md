# 在新机器上跑起来

换电脑后**代码不用重跑，数据集需要重新生成或拷贝**。本文给出三种场景的操作步骤。

---

## 先看清哪些东西能带过去

| 内容 | 是否在仓库里 | 换机器后 |
|---|---|---|
| 全部源码、配置、文档 | ✅ 是 | `git clone` 直接可用 |
| `data/manifest.jsonl`（5 MB 索引） | ✅ 是 | 直接可用，也是复现基准 |
| `data/eval/eval_v1.jsonl`（冻结评测集清单） | ✅ 是 | 直接可用 |
| 三个数据源的图片（约 810 MB） | ❌ 否 | **必须重新生成或拷贝** |
| 训练 checkpoint（`checkpoints/`） | ❌ 否 | 丢失，训练要重跑 |
| 实验报告（`reports/`） | ✅ 是 | 直接可用（历史结果） |

数据集图片不入库是有意的：810 MB 进 git 会让仓库臃肿，而用固定 seed 重新生成
只要约 10 分钟。**复现靠种子，不靠搬运二进制。**

---

## 场景 A：新机器能访问 HuggingFace（推荐）

```bash
# 1. 代码
git clone https://github.com/fyilm/DataEval-QC.git
cd DataEval-QC

# 2. 依赖（GPU 机器必须重装 CUDA 版 PyTorch，仓库默认是 CPU 索引）
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. 数据（约 10 分钟）
bash scripts/run_all.sh --data

# 4. 校验与仓库基准是否一致（关键步骤，别跳过）
uv run python scripts/verify_reproduce.py

# 5. 先做瓶颈诊断，再决定跑什么（见下文）
uv run python -m src.run_exp --mode benchmark
uv run python -m src.run_exp --config configs/tune_gpu.yaml --mode tune
```

## 场景 B：新机器访问 HuggingFace 慢或不通

把本机的 `data/` 整个目录打包拷过去（810 MB，U 盘或网盘均可），
解压覆盖到仓库根目录，然后**跳过第 3 步**，直接从第 4 步开始。

```bash
# 本机打包
tar -czf dataeval-qc-data.tar.gz data/
# 新机器解压到仓库根
tar -xzf dataeval-qc-data.tar.gz
```

注意 `data/` 里包含 `manifest.jsonl`，会和仓库里的同名。解压覆盖后
先跑一次 `verify_reproduce.py` 确认一致 —— 若报不一致，说明拷贝的
数据与仓库基准对不上，此时以**拷贝过来的**为准（后续实验都用它），
但要记一笔：评测集清单若变了需按 `docs/evalset_changelog.md` 登记。

## 场景 C：只想看结论，不跑训练

```bash
git clone https://github.com/fyilm/DataEval-QC.git
cat reports/ablation_table.md        # 消融主表 + 门槛校验
cat reports/vlm_consistency_*.json   # 预标注一致性
```

---

## 关于校验：什么算通过、什么不算

`verify_reproduce.py` 拿 git 里的 `manifest.jsonl` / `eval_v1.jsonl` 当基准，
按 `file + sha1` 比对当前生成的版本。

- **manifest 9700 条全部匹配** → 数据完全复现，此前指标可直接对比。
- **少量长尾样本 sha1 不一致** → 可接受。`build_longtail` 依赖抽样池状态，
  个别长尾图的合成参数会有细微差别，影响面是 200 张里的十几张。
- **缺失或不一致比例超过 5%** → 不可接受。多半是 COCO parquet 分片内容变动
  或采集时网络中断导致样本数不同。此时评测集相当于换了尺子，
  **此前指标不可直接对比**，应按 changelog 规范登记新版本后再跑。

---

## 新机器上先跑什么（重要）

CPU 上的 15 组消融已经跑完，结论是：**合成占比不是瓶颈**。

```
r=0.00 → 0.0000   0.25 → 0.3747   0.50 → 0.4979   0.75 → 0.5481（峰值）   1.00 → 0.5369
```

峰值在 r=0.75，继续堆合成数据反而回落。所以**不要在新机器上重跑占比消融**，
那是已经回答过的问题。直接跑瓶颈诊断：

```bash
uv run python -m src.run_exp --config configs/tune_gpu.yaml --mode tune
```

它会固定 r=0.75，逐个改动分辨率（160→224）、轮数（10→30）、骨干
（mobilenet→efficientnet），跑完直接打印横向对比表，告诉你差距主要来自哪个因素。
定位到主因后，把该设置写进 `configs/ablation.yaml` 再用 3 seed 复现。

---

## 两个已知的待办（GPU 专有）

这两项在 CPU 上做不到，需要在新机器上补：

1. **第三源改真扩散生成** —— 当前 `aigc` 2000 张是编辑变异实现的：
   ```bash
   uv run python -m src.aigc --num 2000 --backend sd
   ```
   之后要重跑去重、切分、评测集与实验。
2. **Qwen2.5-VL 预标注** —— 需 24GB 显存：
   ```bash
   uv run python -m src.label_vlm --backend auto --limit 400
   ```

---

## 环境注意事项（换机器容易踩）

- **CUDA 版 PyTorch**：仓库 `pyproject.toml` 里的 PyTorch 走 CPU 索引，
  GPU 机器上必须重装，否则 `torch.cuda.is_available()` 恒为 False。
- **中文路径**：Linux 上无此问题；若在 Windows 上跑且路径含中文，
  仓库已用 `common.imread()`（`np.fromfile` + `imdecode`）绕过 `cv2.imread` 的限制。
- **DLP 加密**：若新机器也有企业 DLP，`.csv` 会被异步加密。
  规范格式一律用 `.jsonl`，标注结果填 `.jsonl` 或另存为 `.txt`。
- **HF 访问慢**：设 `HF_ENDPOINT=https://hf-mirror.com`，
  或先用 `uv run python scripts/fetch_clip.py` 预下载 CLIP 权重。
