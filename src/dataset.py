"""PyTorch 数据集：多标签质检分类。

设计要点：
- 多标签 multi-hot 目标，BCEWithLogitsLoss 配套
- 增广刻意保守：只有水平翻转 + 极轻的亮度/对比度扰动。
  这里的"缺陷"本身就是低层画质特征，激进的色彩增广会把标签信息一起改掉
  （比如把 over_expo 样本增广回正常曝光），这是质检模型最常见的翻车点。
- 读图统一走 common.imread，绕开 Windows 中文路径问题
"""
from __future__ import annotations

import sys
import pathlib
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import CLASSES, NUM_CLASSES, imread, labels_to_multihot  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class QCTransform:
    """轻量增广。

    注意必须写成类而不是闭包：Windows 上 DataLoader 的 worker 用 spawn 启动，
    闭包无法被 pickle，会报 `Can't get local object 'make_transform.<locals>._t'`。
    """

    def __init__(self, img_size: int = 160, train: bool = True) -> None:
        self.img_size = img_size
        self.train = train
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        import cv2

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.train:
            if np.random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
            # 极轻的亮度/对比度扰动，不破坏缺陷信号
            if np.random.random() < 0.3:
                a = np.random.uniform(0.95, 1.05)
                b = np.random.uniform(-6, 6)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def make_transform(img_size: int = 160, train: bool = True) -> Callable[[np.ndarray], torch.Tensor]:
    return QCTransform(img_size, train)


class QCDataset(Dataset):
    """从 manifest 记录构建的多标签数据集。

    records 需已带 split；可传入采样比例做消融（合成占比控制）。
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        img_size: int = 160,
        train: bool = True,
        sample_ratio: float = 1.0,
        keep_source: str | None = None,
        exclude_source: str | None = None,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        recs = [r for r in records if not r.get("dedup_removed", False)]
        if keep_source:
            recs = [r for r in recs if r["source"] == keep_source]
        if exclude_source:
            recs = [r for r in recs if r["source"] != exclude_source]

        # 按比例子采样。合成占比消融的关键开关。
        if 0.0 < sample_ratio < 1.0 and len(recs) > 1:
            n = max(1, int(round(len(recs) * sample_ratio)))
            idx = rng.choice(len(recs), size=n, replace=False)
            recs = [recs[i] for i in sorted(idx)]

        self.records = recs
        self.transform = make_transform(img_size, train=train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[i]
        img = imread(rec["file"])
        if img is None:
            raise RuntimeError(f"读图失败: {rec['file']}")
        return self.transform(img), torch.tensor(
            labels_to_multihot(rec.get("labels") or []), dtype=torch.float32
        )

    def targets(self) -> np.ndarray:
        return np.array([labels_to_multihot(r.get("labels") or []) for r in self.records], dtype=np.float32)

    @property
    def pos_weight(self) -> torch.Tensor:
        """类别加权：按每类正样本占比计算 pos_weight，缓解类别不均衡。"""
        t = self.targets()
        pos = t.sum(axis=0).clip(min=1)
        neg = len(t) - pos
        return torch.tensor(neg / pos, dtype=torch.float32)
