# 评测集抽样口径（sampling_spec）

> 版本 v1.0 ｜ 冻结日期 2026-09-28
> 本文件说明 eval_v1 的构造口径。评测集冻结后，任何变更必须写进 `evalset_changelog.md`。

## 1. 为什么评测集要冻结

版本对比的前提是口径不变。改评测集而不记录，等于每次考试换卷子还说成绩可比。
所以 eval_v1 一旦生成，只读不改；要改口径就升版本号（eval_v2）并在 changelog 登记。

## 2. 分层网格

分层维度：**缺陷类型 × 场景 × 难度**

| 维度 | 取值 | 格数 |
|---|---|---|
| 缺陷类型 | low_res / blur / over_expo / dark / occlusion / noise / **normal** | 7 |
| 场景 | indoor / outdoor / night | 3 |
| 难度 | easy / mid / hard（normal 层无难度，记 none） | 3（normal 层 1） |

合计 **57 个格子**（6 类 × 3 场景 × 3 难度 = 54，normal × 3 场景 = 3）。

### 难度定义

难度来自合成时的强度档，映射关系固定在 `src/common.py::STRENGTH_TO_DIFFICULTY`：

| 合成强度 | 难度 | 理由 |
|---|---|---|
| mild（轻微） | **hard** | 降级越轻微越难判定 |
| moderate（中度） | mid | — |
| severe（明显） | **easy** | 降级越明显越容易判定 |

多标签样本的难度取"各档中最 mild 的那一档"（最难的主导）。

### 缺陷类型归格

多标签样本按 `params._order` 的**首类**归格，保证每个样本只占一个格子，
避免同一样本在多个格子里被重复计数。归格逻辑见 `src/evalset.py::cell_of`。

## 3. 场景判定（规则化，可复现）

场景由 `src/collect.py::classify_scene` 的确定性规则给出，**不是人工标的**：

```
夜视 night：
    mean_lum < 60
    或 (mean_lum < 95 且 蓝通道均值 > 红通道均值+12 且 蓝通道均值 > 绿通道均值)

室外 outdoor：
    outdoor_score = 天空占比 + 植被占比 > 0.18
    天空占比   = 上 1/3 区域中 (V>150 且 S<70) 的像素比例
    植被占比   = (G 通道 - R 通道 > 20) 的像素比例
    （不满足以上两条则判 indoor）

夜间专项采集的帧强制 scene=night（不走上述规则）。
```

> 局限：这是启发式，室内/室外边界样本会误判。
> 已知局限记录在 data_card.md，不影响冻结（口径本身是确定的，可以复现）。

## 4. 配额分配算法

目标 1200 张分层样本。**等额起步 + 缺额按需回补**：

1. 每个格子目标 = 1200 / 格子数 ≈ 21 张
2. 若某格可用样本 < 目标（典型是夜视格，夜景原图天然稀少），
   取满可用量，缺额让给有余量的格子
3. 循环直到总数 = 1200 或所有格子取满
4. 每格内用固定 seed 随机抽样（`random.Random(42)`）

实现：`src/evalset.py::allocate_quotas`。
实际达成情况见 `reports/evalset_report.json` 的 `cell_stat`。

## 5. 长尾补录 200 张

逆光 / 雨雾 / 污渍 / 拖影 / 极暗+强压缩 五类边缘场景，各 40 张。

- 由 `src/evalset.py::build_longtail` 程序化生成
- **必须来自与 eval 池相同的原图集合**，否则会引入新的原图组造成泄漏
- 标记 `eval_longtail=true`，`review_status=pending_human`
- 当前状态：**程序化补录，待人工复核**。人工复核入口见 README

## 6. 与 train/val/test 的关系

切分顺序（`src/split.py`）：

1. **先按原图分组预留 eval 池**（目标 1400 张，221 个原图组）
2. 剩余原图组按 **7:2:1** 切成 train / val / test

保证：
- eval_v1 与训练**完全隔离**（不同原图组，无泄漏）
- 同一原图的所有变体必在同一 split（`check_leakage` 自检，跨 split 组数必须为 0）
- 保留任务书要求的 7:2:1 口径

## 7. 冻结清单

| 文件 | 作用 |
|---|---|
| `data/eval/eval_v1.jsonl` | **规范格式**，流水线读取此文件 |
| `data/eval/eval_v1.csv` | 人类可读导出 |
| `data/eval/longtail/` | 长尾样本图 |
| `docs/evalset_changelog.md` | 版本变更记录 |
| `reports/evalset_report.json` | 各格达成情况 |

> 注意：本机装有企业 DLP 透明加密，`.csv` 等文档类型会在落地几分钟后被加密。
> 所以规范格式用 `.jsonl`，`.csv` 仅作导出物。推送开源仓库前需检查 CSV 是否被加密。
