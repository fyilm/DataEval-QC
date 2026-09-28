# GPU 规格与所需数量建议

> 本文回答一个问题：**这套流水线要跑完整，需要什么样的显卡、几张、跑多久。**
>
> 文中区分两类数字：
> - **【实测】** —— 本机（Intel Core Ultra 5 225，10 核，32GB RAM，无独显）真实跑出来的；
> - **【推算】** —— 基于实测吞吐做的换算，已注明换算依据与不确定性。
>
> 最后校准时间：见文末"校准记录"。

---

## 1. 先说结论

| 场景 | 显卡规格 | 数量 | 预计耗时 |
|---|---|---|---|
| **最小可行**（只跑分类训练 + 消融 + 校准） | RTX 4090 24GB（或 A100 40GB / RTX 3090） | **1 张** | 约 1–1.5 小时 |
| **推荐配置**（含种子并行，含 VLM 预标注） | RTX 4090 24GB | **2 张** | 约 0.75 小时 + 预标注 1.5 小时 |
| **完整配置**（再含 SDXL 生成第三源） | RTX 4090 24GB | **2 张** | 上述 + 生成约 2–3 小时 |

> 作为对照：**同样 30 组在 CPU 上约需 26 小时**（实测外推，见 §3.3），这也是本仓库
> 拆出 `configs/ablation_cpu.yaml` 只跑单骨干的原因。

- 显存下限由 **Qwen2.5-VL** 决定，不是分类模型：分类模型在 160px/batch64 下 6GB 显存就够。
- **16GB 显存卡（如 RTX 4080 / A4000）也能跑完分类消融**，但跑不了 7B 级别的 VLM 预标注，只能退到 3B 或 CLIP。
- 本流水线**不需要多卡互联（NVLink / InfiniBand）**。每组实验互相独立，多卡是"多进程并行切分任务"，线性加速，用普通 PCIe 机器即可。

---

## 2. 需求拆解：哪些步骤真的需要 GPU

把整条流水线按"是否必须 GPU"拆开，这一步决定了租卡的真实必要性：

| 步骤 | 命令 | 必须 GPU？ | CPU 可行性【实测】 |
|---|---|---|---|
| ① 真实帧采集 | `src.collect` | ❌ 否 | 纯 IO + 解码，1500 张约 3 分钟 |
| ② 程序化降级 | `src.degrade` | ❌ 否 | 纯 OpenCV，6000 张约 4 分钟 |
| ③ 编辑变异（AIGC 降级路线） | `src.aigc --backend edit` | ❌ 否 | 纯 OpenCV，2000 张约 2 分钟 |
| ③′ Stable Diffusion 生成 | `src.aigc --backend sd` | ✅ **是** | CPU 上不现实（单图分钟级） |
| ④ pHash 去重 | `src.dedup` | ❌ 否 | 9500 张约 8 秒 |
| ⑤ 切分 / 评测集冻结 | `src.split` `src.evalset` | ❌ 否 | 秒级 |
| ⑥ **分类训练 + 消融** | `src.run_exp` | ⚠️ 强烈建议 | CPU 可跑，但见 §3 |
| ⑦ 置信度校准 | `src.calibrate` | ❌ 否 | 只在 logits 上做优化，秒级 |
| ⑧ **CLIP 预标注** | `src.label_vlm --backend clip` | ⚠️ 建议 | CPU 约 1.5 img/s，400 张约 4 分钟 |
| ⑧′ **Qwen2.5-VL 预标注** | `--backend qwen_local` | ✅ **是** | 7B 模型 CPU 上不可用 |

结论：**只有 ③′（SD 生成）、⑥（训练，为了时间）、⑧′（VLM 预标注）三处真正需要租卡。**
其余步骤在你本机就能跑完，不需要为它们付费。

---

## 3. 分类训练：为什么值得租卡

### 3.1 实测吞吐基线

`src.run_exp --mode benchmark` 在**本机 CPU** 上的结果【实测】：

| 模型 | img_size | batch | 吞吐（推理） |
|---|---|---|---|
| mobilenet_v3_small | 160 | 64 | **239 img/s** |
| efficientnet_b0 | 160 | 64 | **40.8 img/s** |

注意这是**推理**吞吐。训练含前向 + 反向 + 优化器，经验上吞吐约为推理的 **1/3**，即 CPU 上约 **80 img/s（mobilenet）/ 13 img/s（efficientnet）**。

### 3.2 实际工作量

消融实验配置：2 模型 × 5 个合成占比 × 3 seed = **30 组**，每组 10 epochs。

训练集规模随合成占比 r 变化（真实侧固定为 train split 全部真实帧 895 张）【实测】：

| r | 真实帧 | 合成帧 | 合计 |
|---|---|---|---|
| 0.00 | 895 | 0 | 895 |
| 0.25 | 895 | 298 | 1193 |
| 0.50 | 895 | 895 | 1790 |
| 0.75 | 895 | 2685 | 3580 |
| 1.00 | 0 | 3580 | 3580 |

平均每组约 **2200 张 × 10 epochs = 22000 次样本前向反向**。

- CPU（mobilenet，80 img/s 训练）：22000 / 80 ≈ **275 s/组**，30 组约 **2.3 小时**
- CPU（efficientnet，13 img/s 训练）：22000 / 13 ≈ **1690 s/组**，15 组约 **7 小时**

两者合计 **CPU 全量约 9 小时**。这就是我最终决定"跑全量不砍档"但仍需要 GPU 的原因 —— 换卡后这个时间能压到 1 小时内。

### 3.3 CPU 实测耗时（原始依据）

`r=0` 组实测【实测】：895 张 × 10 epochs = 8950 samples，`train_seconds = 375.8`，
即 **训练吞吐约 23.8 samples/s**。

注意这比 §3.1 的推理吞吐（239 img/s）低一个数量级，不只是"反向传播约 3 倍"的经验值 ——
160px 小图下 DataLoader 解码与增广的 CPU 开销占了大头。用推理吞吐直接估训练时间会严重低估。

按各档训练集规模外推，每 seed 的样本量：

| r | 训练集 | 每 seed 样本量（×10 epochs） |
|---|---|---|
| 0.00 | 895 | 8950 |
| 0.25 | 1193 | 11930 |
| 0.50 | 1790 | 17900 |
| 0.75 | 3580 | 35800 |
| 1.00 | 3580 | 35800 |
| **合计/seed** | | **110380** |

30 组 = 3 seed × 2 模型 = **331140 samples**。

- mobilenet（23.8 samples/s）：331140 / 2 / 23.8 ≈ **3.9 小时**（两个模型各占一半）
- efficientnet（推理慢 5.86 倍，训练同比例外推 ≈ 4.06 samples/s）：**约 22.7 小时**
- **CPU 全量合计约 26 小时**

### 3.4 换到 GPU 后的推算

小模型 + 小分辨率场景下，RTX 4090 相对这款 CPU 的经验加速比约 **15–25 倍**。
注意 160px 下 GPU 利用率偏低，数据加载可能成为瓶颈，实际可能落在 10–20 倍。

| 模型 | CPU 耗时【实测外推】 | GPU 单卡推算 | 2 卡并行推算 |
|---|---|---|---|
| mobilenet_v3_small（15 组） | 3.9 h | 约 10–23 分钟 | 约 5–12 分钟 |
| efficientnet_b0（15 组） | 22.7 h | 约 55–135 分钟 | 约 28–68 分钟 |
| **合计 30 组** | **26.6 h** | **约 1–2.6 小时** | **约 0.5–1.3 小时** |

**推算依据与不确定性：** 加速比是经验值，未在本机实测（本机无 NVIDIA 显卡）。
§1 表格里的"1–1.5 小时"取的是偏保守的中间值。若 GPU 机器上数据加载成为瓶颈，
可把 `batch_size` 调大或 `num_workers` 调高来缓解。

> 在真正跑过 GPU 之前，以上均为推算；跑完后请回填 §8 校准记录，把推算值替换为实测值。

---

## 4. 显存需求逐项核算

| 任务 | 模型 | 精度 | 峰值显存 | 说明 |
|---|---|---|---|---|
| 分类训练 | mobilenet_v3_small @160, bs64 | fp32 | **约 3 GB** | 参数 2.5M，activation 占大头 |
| 分类训练 | efficientnet_b0 @160, bs64 | fp32 | **约 5 GB** | 参数 5.3M |
| 分类训练 | 同上但 @224, bs64 | fp32 | 约 10 GB | 若想提精度可上调分辨率 |
| VLM 预标注 | Qwen2.5-VL-3B | fp16 | **约 8–10 GB** | 权重 6GB + KV cache |
| VLM 预标注 | Qwen2.5-VL-7B | fp16 | **约 18–22 GB** | 权重 14GB + KV cache |
| AIGC 生成 | SD 1.5 | fp16 | 约 4 GB | 512×512 |
| AIGC 生成 | SDXL | fp16 | 约 10–12 GB | 1024×1024 |

**关键判断：显存瓶颈不在分类，在 VLM。**

- 只跑分类消融 → **8GB 显存就够**（GTX 4060 8G / RTX 2000 Ada 都行）
- 要跑 Qwen2.5-VL-7B → **必须 24GB**（RTX 3090 / 4090 / A100 / L20）
- 退一步用 3B → **12GB 够**（RTX 3060 12G / 4070）

---

## 5. 数量：为什么推荐 2 张而不是 8 张

本流水线没有分布式训练需求 —— 30 组实验是**互相独立的单卡任务**，不存在梯度同步，因此：

- 多卡的加速是**任务级并行**，近似线性（30 组 ÷ N 张卡）
- **不需要 NVLink / NVSwitch / RDMA**，机器上插几张普通 PCIe 卡即可
- 卡数超过 **4 张**后收益骤降：只剩 30 / 4 ≈ 7.5 组/卡，任务调度和数据加载的开销占比上升，且小模型单卡占用率低，容易"卡在等数据"

推荐 **2 张**：把 3 个 seed 拆开，seed 42/43 走卡 0、seed 44 走卡 1（或按组号奇偶切分），实现接近 2× 加速，成本只加一倍。

如果需要同时跑 VLM 预标注，可以让第 2 张卡专职做预标注 —— 这样两条线互不抢占。

---

## 6. 具体建议（可直接照此下单）

### 方案 A：最小可行（推荐先用这个）

```
显卡：1 × RTX 4090 24GB（或 A100 40GB / L20 / RTX 3090）
系统：Ubuntu 22.04 + CUDA 12.1+
CPU：8 核以上（数据加载用，别太寒酸）
内存：32 GB
硬盘：100 GB 可用（数据集约 2.5GB + 权重缓存 + checkpoint）
租期：4 小时足够跑完分类全量 + 校准 + 3B VLM 预标注
```

跑完拿到的东西：30 组消融主表 + 校准前后 ECE 对比 + 可靠性图 + VLM 一致率报告。

### 方案 B：完整版（要跑 SD 第三源 + 7B VLM）

```
显卡：2 × RTX 4090 24GB
其余同方案 A
租期：8 小时
```

比 A 多做：`src.aigc --backend sd` 真实扩散生成（约 2000 张，1024px 约 3–4 秒/张 → 单卡约 2.5 小时）、Qwen2.5-VL-7B 全量预标注（9500 张 × 约 1.2 s/张 → 单卡约 3 小时，双卡约 1.5 小时）。

### 不选的方案

- **A100 80GB**：显存严重过剩，单价贵 2–3 倍，本任务用不到。除非平台只剩这种卡。
- **多机多卡**：没有任何必要，30 组独立小任务用不上分布式。
- **消费级 8GB 卡跑 7B VLM**：会 OOM，别试。

### 成本

各平台（AutoDL / 恒源云 / 揽睿星舟 / 腾讯云 GPU / 阿里云）报价随供需浮动较大，**请以平台实时报价为准**。按常见区间粗估：RTX 4090 约 ¥2–5/小时，则方案 A（1 卡 × 4 小时）约 **¥8–20**，方案 B（2 卡 × 8 小时）约 **¥32–80**。整体是个几十块钱量级的开销。

---

## 7. GPU 机器上的执行清单

租到卡后按这个顺序跑（已在 README 的一键复现里给出，此处是 GPU 版）：

```bash
# 0. 环境：CUDA 版 PyTorch（注意与本机 CPU 索引的区别）
uv sync --extra vlm
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 1. 先确认卡被认到
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. 数据（CPU 部分，10 分钟内跑完）
uv run python -m src.collect  --num 1200 --night-quota 300
uv run python -m src.degrade  --num 6000
uv run python -m src.aigc    --num 2000 --backend edit     # 或 --backend sd 走真实扩散
uv run python -m src.dedup
uv run python -m src.split
uv run python -m src.evalset

# 3. 吞吐基准（先跑，用它定档）
uv run python -m src.run_exp --mode benchmark

# 4. 全量消融（30 组）
uv run python -m src.run_exp --config configs/ablation.yaml --mode grid
#   双卡并行：按 seed 拆开，各自写同一份 results，已完成的会自动跳过
#   CUDA_VISIBLE_DEVICES=0 uv run python -m src.run_exp --config configs/ablation.yaml --mode grid &
#   CUDA_VISIBLE_DEVICES=1 uv run python -m src.run_exp --config configs/ablation.yaml --mode grid &
#   wait

# 4'. CPU 上请改用单骨干配置（15 组，约 4 小时；跨模型对比留给 GPU）
#   uv run python -m src.run_exp --config configs/ablation_cpu.yaml --mode grid

# 5. 校准（在最佳档位上做）
uv run python -m src.calibrate --tag best

# 6. VLM 预标注（需 24GB；12GB 卡请改用 --backend clip）
uv run python -m src.label_vlm --backend auto --limit 400
```

**避坑提示（本机踩过的）：**
- 本机 `pyproject.toml` 里 PyTorch 走的是 **CPU 索引**。GPU 机器上必须重装 CUDA 版，否则 `torch.cuda.is_available()` 恒为 False。
- 路径含中文时 Windows 上 `cv2.imread` 会静默返回 None；Linux 无此问题，仓库里的 `imread()` 封装两种系统都安全。
- 若平台出口访问 HuggingFace 慢，设 `HF_ENDPOINT=https://hf-mirror.com`，或先把权重下到 `.cache/clip` 再用。

---

## 8. 校准记录

| 日期 | 事件 | 影响 |
|---|---|---|
| 2026-09-28 | 本机 CPU 实测推理吞吐（mobilenet 239 / efficientnet 40.8 img/s） | §3.1 数字 |
| 2026-09-28 | 实测训练吞吐 23.8 samples/s（r=0 组 8950 samples / 375.8 s） | §3.3 数字；据此发现"用推理吞吐估训练"会低估 10 倍 |
| 2026-09-28 | 据此外推 CPU 全量 30 组约 26 小时，判定不可行 | 拆出 `configs/ablation_cpu.yaml`；§1 时间改为 1–1.5 小时 |
| 待补充 | GPU 机器上实测 30 组真实耗时 | 把 §3.4 的推算替换为实测，§1 表格改为【实测】 |

> 在拿到 CPU 全量真实耗时前，§1 的"0.5–1 小时"是**保守推算**，不是承诺值。
