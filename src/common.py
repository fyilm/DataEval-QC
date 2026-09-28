"""公共常量、路径与工具函数。

整个项目的数据地基：六类缺陷定义、样本记录 schema、seed 控制、哈希登记。
所有模块都从这里导入，避免出现两份不一致的类名表。
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
from typing import Any, Iterable

import numpy as np
import torch

# ---------------------------------------------------------------- 路径
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REAL_DIR = DATA_DIR / "real"
PROC_DIR = DATA_DIR / "procedural"
AIGC_DIR = DATA_DIR / "aigc"
EVAL_DIR = DATA_DIR / "eval"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = REPORTS_DIR / "figures"

MANIFEST_PATH = DATA_DIR / "manifest.jsonl"

# ---------------------------------------------------------------- 标签体系
CLASSES = ["low_res", "blur", "over_expo", "dark", "occlusion", "noise"]

CLASS_CN = {
    "low_res": "低分辨率",
    "blur": "模糊",
    "over_expo": "过曝",
    "dark": "欠曝/黑屏",
    "occlusion": "遮挡/污渍",
    "noise": "压缩噪声",
}

CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

SOURCES = ["real", "procedural", "aigc"]
SPLITS = ["train", "val", "test"]
SCENES = ["indoor", "outdoor", "night"]
DIFFICULTIES = ["easy", "mid", "hard"]

# 合成强度 -> 难度：降级越轻微越难判定，越明显越容易判定
STRENGTH_TO_DIFFICULTY = {"mild": "hard", "moderate": "mid", "severe": "easy"}


def labels_to_multihot(labels: Iterable[str]) -> list[int]:
    """类别名列表 -> 6 维 multi-hot 向量。"""
    vec = [0] * NUM_CLASSES
    for lab in labels:
        if lab in CLASS_IDX:
            vec[CLASS_IDX[lab]] = 1
    return vec


def multihot_to_labels(vec: Iterable[float]) -> list[str]:
    return [CLASSES[i] for i, v in enumerate(vec) if v and float(v) > 0.5]


# ---------------------------------------------------------------- 设备
def get_device(prefer: str | None = None) -> torch.device:
    """自动选择设备。可在配置里强制指定 cpu / cuda。"""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict[str, Any]:
    dev = get_device()
    info: dict[str, Any] = {
        "device": str(dev),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def suggest_num_workers(device: torch.device) -> int:
    """CPU 上给足 worker，GPU 上适度，Windows 上避免过多进程开销。"""
    cpu = os.cpu_count() or 4
    if device.type == "cuda":
        return min(8, cpu)
    return max(0, min(8, cpu - 2))


# ---------------------------------------------------------------- 随机性
def set_seed(seed: int) -> None:
    """固定所有随机源。实验可复现的唯一入口。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- 哈希
def sha1_of_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_of_file(path: str | pathlib.Path) -> str:
    p = pathlib.Path(path)
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- manifest
def ensure_dirs() -> None:
    for d in (
        DATA_DIR,
        REAL_DIR,
        PROC_DIR,
        AIGC_DIR,
        EVAL_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        CONFIGS_DIR,
        DOCS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def append_manifest(records: list[dict[str, Any]], path: pathlib.Path = MANIFEST_PATH) -> None:
    """逐条追加登记。每生成一张样本即调用，天然支持增量与回滚。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def reset_manifest(path: pathlib.Path = MANIFEST_PATH) -> None:
    if path.exists():
        path.unlink()


def export_manifest_json(
    records: list[dict[str, Any]], out: pathlib.Path = DATA_DIR / "manifest.json"
) -> pathlib.Path:
    """导出为单文件 JSON 数组，便于人工查看与版本比对（任务书 §7.2 的 manifest.json）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return out


def write_json(obj: Any, path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p


def read_json(path: str | pathlib.Path) -> Any:
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: str | pathlib.Path) -> str:
    """统一存相对路径，保证仓库换机器后 manifest 仍可用。"""
    try:
        return pathlib.Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return pathlib.Path(path).as_posix()


# ---------------------------------------------------------------- 图像读写
def imread(path: str | pathlib.Path) -> "np.ndarray | None":
    """读图。必须走这个封装，不能直接用 cv2.imread。

    Windows 上 cv2.imread 不支持非 ASCII 路径（本仓库路径含中文），
    会静默返回 None。改用 np.fromfile + imdecode 绕过。
    """
    import cv2

    p = pathlib.Path(path)
    if not p.exists():
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite(path: str | pathlib.Path, img: "np.ndarray", params: list | None = None) -> bool:
    """写图。同样绕开 cv2.imwrite 的非 ASCII 路径问题。"""
    import cv2

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix or ".jpg"
    ok, buf = cv2.imencode(suffix, img, params or [])
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def encode_jpeg(img: "np.ndarray", quality: int = 92) -> bytes:
    """编码为 JPEG 字节，用于算 sha1 与落盘。"""
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


# ---------------------------------------------------------------- 画质指标
def assess_quality(img: "np.ndarray") -> dict[str, float]:
    """客观画质指标，用于筛查、难度校验与数据卡统计。"""
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "mean_lum": float(gray.mean()),
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "width": float(img.shape[1]),
        "height": float(img.shape[0]),
    }
