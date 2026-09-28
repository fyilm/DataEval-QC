# DataEval-QC：合成数据驱动的视觉质检数据集与评测体系

从零构建"**真实帧 + 程序化合成 + AIGC 生成**"三源混合的视觉质检数据集，配套按分层抽样口径冻结的评测集、置信度校准实验与合成数据占比消融实验。

六类画质缺陷的多标签分类：`low_res`（低分辨率）、`blur`（模糊）、`over_expo`（过曝）、`dark`（欠曝/黑屏）、`occlusion`（遮挡/污渍）、`noise`（压缩噪声）。

**当前状态**：数据集已真实产出（9700 张），全链路代码可运行，消融实验进行中。

---

## 快速开始

```bash
# 1. 装依赖（用 uv；会自动按 pyproject.toml 建虚拟环境）
uv sync

# 2. 只跑数据管线（约 10 分钟）
bash scripts/run_all.sh --data

# 3. 跑冒烟验证（1 epoch，确认全链路通）
uv run python -m src.run_exp --config configs/smoke.yaml --mode single

# 4. 全量（含 30 组消融）
bash scripts/run_all.sh
```

依赖用 **uv** 管理，版本用 **git** 管理。PyTorch 走 CPU 索引（`pyproject.toml` 里配了 `pytorch-cpu`），避免默认拉 2.5GB 的 CUDA 轮子；GPU 机器上需另外重装 CUDA 版，见 [`docs/gpu_plan.md`](docs/gpu_plan.md)。

---

## 流水线

```
src.collect    采集真实清晰帧（HuggingFace COCO parquet）+ 质量筛查
     │
     ├─→ src.degrade    程序化降级（三档强度）      → procedural 源
     ├─→ src.aigc       编辑变异 / 扩散生成         → aigc 源
     │
src.dedup      pHash 跨源去重（只在异母图之间比）
src.split      按 origin_id 分组切分 + 泄漏自检
src.evalset    分层抽样 1200 + 长尾 200 → eval_v1.jsonl 冻结
     │
src.run_exp    训练 / 吞吐基准 / 消融（合成占比 × 模型 × seed）
src.calibrate  温度标定 + 可靠性图
src.label_vlm  预标注一致性核对（rule / CLIP / Qwen 四后端可插拔）
```

---

## 验收对照

对照任务书的验收项，如实标注当前状态（✅ 已完成 / ⏳ 进行中或待输入）：

| # | 验收项 | 要求 | 当前状态 |
|---|---|---|---|
| 1 | 数据规模 | ≥ 8000 张 | ✅ **9700 张**（real 1500 / procedural 6200 / aigc 2000） |
| 2 | pHash 跨源去重 | 需实现并报告 | ✅ 已跑，跨源近似对 0、剔除 0（口径见下文） |
| 3 | 防泄漏 | 同原图变体同 split | ✅ 自检通过，1500 母图组跨 split **0** 个 |
| 4 | 评测集冻结 | 文件 + 口径文档 + changelog | ✅ `eval_v1.jsonl` 1400 张 + [`sampling_spec.md`](docs/sampling_spec.md) + [`evalset_changelog.md`](docs/evalset_changelog.md) |
| 5 | 一键复现 | 固定 seed ×3 | ✅ `scripts/run_all.sh`，seed 42/43/44；⏳ 30 组跑完回填主表 |
| 6 | Cohen's Kappa | ≥ 0.85 | ⏳ `src.agreement` 已实现（多标签逐类 Kappa 后 macro），**待双人标注 CSV 输入** |
| 7 | 模型指标 | macro-F1 ≥ 0.85、模糊/低清召回 ≥ 0.95 | ⏳ 待消融实验主表（smoke 1 epoch 已跑通：macro-F1 0.41） |
| 8 | 置信度校准 | ECE 前后对比 + 可靠性图 | ✅ 已跑通：ECE 0.5985 → 0.5254，Brier 0.1897 → 0.185，T=1.93 |
| 9 | 开源仓库 | GitHub 公开 | ⏳ 代码与文档就绪，待推送 |

---

## 关键设计决策

这几处是任务书没写清、需要自己定口径的地方，也是最容易踩坑的地方：

**1. 去重只在异母图之间比对**
合成样本与其母图天然高度相似，全局做 pHash 比对会把合成数据整批误删。因此只在 `origin_id` 不同的样本对之间比对，15550 对同母图样本按设计保留。

**2. 按母图分组切分，而不是按样本随机切分**
同一母图的"原图 + 低清版 + 模糊版"若被分到 train 和 test，模型靠记住母图内容就能作弊。切分以 `origin_id` 为单位，并输出 `leakage_check` 自检。

**3. 多标签 ECE 要改口径**
任务书 §7.4 给的 ECE 公式是单标签的 `probs.max(1)`，对多标签不适用。这里把 (样本, 类别) 摊平成二元判定再分箱；单标签版本保留为对照。Brier 同理做逐类 macro。

**4. 消融里"合成占比"的定义**
任务书说"总样本量固定"，但真实帧只有 1500 张，凑不出 5600 张真实帧。这里定义为 **r = 训练集中合成缺陷帧的占比**：真实侧固定为 train split 全部真实帧（895 张），合成侧按 `k = len(real)·r/(1-r)` 反推。代价是训练总量随 r 变化（895 → 3580），曲线左端同时包含"数据量不足"的效应 —— 这一点在报告里明确标注，不掩盖。

> ⚠ **r 有双重含义，读曲线时必须注意**：本设计中真实帧全部是**干净帧**（作为负样本与母图），
> 所以 r 既是"合成数据占比"，也是"正样本占比"。因此 **r=0 等于零正样本**，模型学不到任何缺陷，
> macro-F1 必然为 0（实测确实如此）—— 它是退化对照点，不是"纯真实数据基线"。
> 若要把两个因素解耦，需要一部分带缺陷标注的真实帧，当前数据集不具备这个条件。

**5. 阈值按召回约束搜索，不按 F1**
质检场景漏检代价高于误检，阈值搜索在 `recall_floor=0.95` 约束下最大化精确率，而不是取 F1 最优点。

**6. 难度由强度反推**
`mild → hard`、`moderate → mid`、`severe → easy`。强度越弱，缺陷越难识别。

---

## GPU 规格建议

整套流水线的主力工作（采集、降级、去重、切分）在 CPU 上就能跑完，**只有三处真正需要显卡**：扩散生成第三源、分类训练加速、Qwen2.5-VL 预标注。

| 场景 | 规格 | 数量 | 预计耗时 |
|---|---|---|---|
| 最小可行（分类训练 + 消融 + 校准） | RTX 4090 24GB | **1 张** | 约 1–1.5 小时 |
| 推荐（含 VLM 预标注 + 种子并行） | RTX 4090 24GB | **2 张** | 约 2.5 小时 |
| 完整（再含 SD 扩散生成） | RTX 4090 24GB | **2 张** | 约 5 小时 |

作为对照：**同样 30 组在 CPU 上约需 26 小时**（实测外推）。这也是仓库里额外提供
`configs/ablation_cpu.yaml`（只跑单骨干，约 4 小时）的原因 —— CPU 上先拿到完整占比曲线，
跨模型对比留给 GPU。

要点：显存瓶颈在 VLM（7B 需 24GB）而非分类模型（6GB 够）；**不需要多卡互联**，30 组实验互相独立，多卡是任务级并行；超过 4 张卡收益骤降。

完整核算（逐项显存、吞吐实测、执行清单、避坑提示）见 [`docs/gpu_plan.md`](docs/gpu_plan.md)。

---

## 目录

```
configs/    消融与冒烟配置
data/       manifest.jsonl（唯一索引）、eval/eval_v1.jsonl（冻结评测集）
docs/       标注规范 / 抽样口径 / 数据卡 / 评测集变更 / GPU 方案
reports/    各阶段报告与图（dedup / split / evalset / calibration / ablation）
scripts/    一键复现
src/        流水线源码（见上文）
```

文档索引：

- [`docs/annotation_spec.md`](docs/annotation_spec.md) —— 六类缺陷判定边界、冲突优先级链
- [`docs/sampling_spec.md`](docs/sampling_spec.md) —— 57 格分层网格、配额算法、冻结流程
- [`docs/data_card.md`](docs/data_card.md) —— 数据集构成与**已知偏差**
- [`docs/evalset_changelog.md`](docs/evalset_changelog.md) —— 评测集变更历史
- [`docs/gpu_plan.md`](docs/gpu_plan.md) —— GPU 规格与数量建议

---

## 已知限制

诚实列出，这些都会写进最终报告：

- **域差距**：真实帧来自 COCO 自然图像而非监控画面，缺少鱼眼畸变、红外灰阶、OSD 叠加等监控特有退化。**本数据集指标不等同于真实产线性能。**
- **第三源未走扩散**：当前 aigc 2000 张是编辑变异实现（本机无 NVIDIA 显卡，SD 在 CPU 上不现实）。租到卡后应重跑 `--backend sd` 再评估。
- **标签是构造标签**：反映"施加了什么降级"，不反映"人眼是否认为有缺陷"。这正是要做一致性核对的原因。
- **场景不均**：outdoor 64.6% vs indoor 15.1%。
- **干净基线偏高**：采集时已用 `lap_var ≥ 80` 筛过一遍，模型可能学不会容忍真实场景的轻微退化。

---

## 环境注意事项

- **中文路径**：Windows 上 `cv2.imread` 不支持非 ASCII 路径会静默返回 None。仓库统一用 `common.imread()`（`np.fromfile` + `imdecode`）绕过。
- **DataLoader spawn**：Windows 下 transform 必须是可 pickle 的类，不能是闭包。见 `dataset.QCTransform`。
- **企业 DLP 加密**：部分环境会按后缀异步加密 `.csv`/`.xls`/`.doc`（文件头变 `%TSD-Header-###%`）。因此规范格式用 `.jsonl` 和 `.json`，CSV 只作导出物。
- **CLIP 权重**：`openai/clip-vit-base-patch32` 在 HF 上只有 `pytorch_model.bin`（无 safetensors），HF 缓存偶发 0 字节损坏。可预先下到 `.cache/clip` 后用 `CLIP_LOCAL_DIR` 指定。
