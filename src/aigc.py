"""第三数据源：AIGC 生成 / 图像编辑变异。

后端按设备自动选择：
    sd   —— Stable Diffusion(+ControlNet) 文生图，需要 NVIDIA 显卡，走 diffusers
    edit —— 图像编辑变异，纯 CPU 可跑，无需显卡（本机默认走这条）

为什么编辑变异能算"第三源"：它的降级机制与 degrade.py 完全不同
（传感噪声模型、雾雨模型、色散模型 vs 直接的下采样/gamma/JPEG），
因此产生的低层纹理指纹不同 —— 这正是"合成数据分布偏移"可被检验的前提。

编辑变异同样产出六类标签，另含约 15% 的"难负例"：
做了轻度编辑但不构成缺陷，标签为空，用来压误检。
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    AIGC_DIR,
    CLASSES,
    STRENGTH_TO_DIFFICULTY,
    ensure_dirs,
)
from src.degrade import save_derived  # noqa: E402
from src.common import imread  # noqa: E402

STRENGTHS = ["mild", "moderate", "severe"]
CLEAN_PROB = 0.15  # 难负例比例


# ---------------------------------------------------------------- 编辑变异实现


def _poisson_gaussian(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """传感噪声：泊松-高斯模型，与 JPEG 块效应纹理不同。"""
    scale = {"mild": (0.004, 0.012), "moderate": (0.015, 0.035), "severe": (0.04, 0.09)}[strength]
    f = img.astype(np.float32) / 255.0
    shot = np.random.poisson(np.clip(f, 0, 1) * 255.0 * rng.uniform(*scale)) / 255.0
    read_sigma = {"mild": 0.004, "moderate": 0.012, "severe": 0.030}[strength]
    read = np.random.normal(0, read_sigma, f.shape).astype(np.float32)
    out = np.clip(f + shot + read, 0, 1)
    return (out * 255).astype(np.uint8)


def _fog(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """雾/霾：亮度提升 + 对比度下降，模拟大气散射。"""
    density = {"mild": 0.15, "moderate": 0.35, "severe": 0.6}[strength] * rng.uniform(0.8, 1.2)
    h, w = img.shape[:2]
    base = np.full((h, w, 3), rng.randint(200, 245), dtype=np.float32)
    # 低频扰动，避免成为纯色薄膜
    noise = np.random.normal(0, 12, (max(2, h // 16), max(2, w // 16), 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    base = np.clip(base + noise, 0, 255)
    out = img.astype(np.float32) * (1 - density) + base * density
    return np.clip(out, 0, 255).astype(np.uint8)


def _rain(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """雨条纹：斜向细线，带轻微模糊。"""
    out = img.copy()
    n = {"mild": 40, "moderate": 120, "severe": 260}[strength]
    n = int(n * rng.uniform(0.7, 1.3))
    h, w = img.shape[:2]
    angle = rng.uniform(-0.45, -0.15)
    for _ in range(n):
        x0 = rng.randint(-w // 4, w)
        y0 = rng.randint(0, h)
        length = rng.randint(int(h * 0.03), int(h * 0.14))
        x1 = int(x0 + length * np.sin(angle))
        y1 = int(y0 + length * np.cos(angle))
        v = rng.randint(150, 235)
        cv2.line(out, (x0, y0), (x1, y1), (v, v, v), 1, cv2.LINE_AA)
    alpha = {"mild": 0.25, "moderate": 0.45, "severe": 0.7}[strength]
    return cv2.addWeighted(out, alpha, img, 1 - alpha, 0)


def _chromatic_aberration(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """色散：R/B 通道错位，镜头畸变常见现象。"""
    shift = {"mild": 1, "moderate": 2, "severe": 4}[strength]
    shift = max(1, int(shift * rng.uniform(0.8, 1.2)))
    b, g, r = cv2.split(img)
    m = np.float32([[1, 0, shift], [0, 1, 0]])
    r = cv2.warpAffine(r, m, (img.shape[1], img.shape[0]))
    m = np.float32([[1, 0, -shift], [0, 1, 0]])
    b = cv2.warpAffine(b, m, (img.shape[1], img.shape[0]))
    return cv2.merge([b, g, r])


def _mud_splatter(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """泥点/污渍：不规则半透明斑块。"""
    out = img.copy()
    n = {"mild": 1, "moderate": 3, "severe": 6}[strength]
    h, w = img.shape[:2]
    for _ in range(n):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rad = int(min(h, w) * {"mild": 0.05, "moderate": 0.12, "severe": 0.22}[strength] * rng.uniform(0.7, 1.3))
        rad = max(4, rad)
        color = (rng.randint(30, 90), rng.randint(25, 80), rng.randint(20, 70))
        cv2.circle(out, (cx, cy), rad, color, -1)
        for _ in range(rng.randint(3, 8)):  # 卫星小点
            d = rng.randint(rad, rad * 3)
            a = rng.uniform(0, 2 * np.pi)
            px = int(np.clip(cx + d * np.cos(a), 0, w - 1))
            py = int(np.clip(cy + d * np.sin(a), 0, h - 1))
            cv2.circle(out, (px, py), max(1, rad // 6), color, -1)
    alpha = {"mild": 0.35, "moderate": 0.6, "severe": 0.85}[strength]
    return cv2.addWeighted(out, alpha, img, 1 - alpha, 0)


def _vignette_dark(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """欠曝：暗角 + 整体压暗，与 gamma 曲线的指纹不同。"""
    h, w = img.shape[:2]
    strength_k = {"mild": 0.55, "moderate": 0.85, "severe": 1.25}[strength]
    x = np.linspace(-1, 1, w)[None, :]
    y = np.linspace(-1, 1, h)[:, None]
    r = np.sqrt(x**2 + y**2)
    mask = np.clip(1 - strength_k * r**2, 0, 1).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w // 12, sigmaY=h // 12)[:, :, None]
    out = img.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def _bloom(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """过曝：高光泛光（bloom），与 gamma 提亮的指纹不同。"""
    thresh = {"mild": 225, "moderate": 205, "severe": 180}[strength]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    k = {"mild": 15, "moderate": 31, "severe": 61}[strength]
    glow = cv2.GaussianBlur(bright.astype(np.float32), (k, k), 0)
    glow = glow / (glow.max() + 1e-6) * 255.0
    amount = {"mild": 0.25, "moderate": 0.45, "severe": 0.7}[strength]
    out = img.astype(np.float32) + glow[:, :, None] * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def _resample_lowres(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """低分辨率：反复重采样 + 锐化回补，与单次下采样放大指纹不同。"""
    h, w = img.shape[:2]
    scale = {"mild": rng.uniform(1.4, 1.9), "moderate": rng.uniform(2.0, 3.0), "severe": rng.uniform(3.2, 5.0)}[
        strength
    ]
    out = img
    times = {"mild": 2, "moderate": 3, "severe": 4}[strength]
    for _ in range(times):
        small = cv2.resize(out, (max(1, int(w / scale)), max(1, int(h / scale))), interpolation=cv2.INTER_NEAREST)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if strength == "mild":  # 轻度档回补锐化，制造"看着还行但细节已丢"的难例
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _defocus_blur(img: np.ndarray, rng: random.Random, strength: str) -> np.ndarray:
    """失焦：圆盘卷积核 / 中值滤波，与高斯和运动模糊指纹不同。"""
    if rng.random() < 0.5:
        r = {"mild": 2, "moderate": 4, "severe": 7}[strength]
        r = max(1, int(r * rng.uniform(0.8, 1.2)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)).astype(np.float32)
        k /= k.sum()
        return cv2.filter2D(img, -1, k)
    k = {"mild": 3, "moderate": 5, "severe": 9}[strength]
    k = max(3, k | 1)
    return cv2.medianBlur(img, k)


def _clean_edit(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """难负例：轻度编辑但不构成缺陷（白平衡/对比度/轻微锐化）。"""
    out = img.astype(np.float32)
    mode = rng.choice(["wb", "contrast", "sharpen", "sat"])
    if mode == "wb":
        gains = np.array([rng.uniform(0.95, 1.05) for _ in range(3)], dtype=np.float32)
        out = out * gains[None, None, :]
    elif mode == "contrast":
        out = (out - 127.5) * rng.uniform(0.92, 1.10) + 127.5
    elif mode == "sharpen":
        blur = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.1))
        out = cv2.addWeighted(img.astype(np.float32), 1.5, blur.astype(np.float32), -0.5, 0)
    else:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.85, 1.18)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


EDIT_FUNCS = {
    "low_res": _resample_lowres,
    "blur": _defocus_blur,
    "over_expo": _bloom,
    "dark": _vignette_dark,
    "occlusion": lambda img, rng, st: (
        _rain(img, rng, st) if rng.random() < 0.5 else (_fog(img, rng, st) if rng.random() < 0.5 else _mud_splatter(img, rng, st))
    ),
    "noise": _poisson_gaussian,
}


def edit_variation(
    img: np.ndarray, rng: random.Random, classes: list[str] | None = None
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """产出一张编辑变异样本。约 15% 为难负例（标签为空）。"""
    if classes is None:
        if rng.random() < CLEAN_PROB:
            out = _clean_edit(img, rng)
            return out, [], {"_difficulty": "none", "_order": [], "edit": "clean_negative"}
        n = 2 if rng.random() < 0.2 else 1
        classes = rng.sample(CLASSES, n)

    order = classes[:]
    rng.shuffle(order)
    out = img
    params: dict[str, Any] = {}
    for c in order:
        st = rng.choice(STRENGTHS)
        out = EDIT_FUNCS[c](out, rng, st)
        params[c] = {"strength": st}
    if rng.random() < 0.35:  # 叠加色散，增加真实感
        out = _chromatic_aberration(out, rng, rng.choice(STRENGTHS))
        params["chromatic"] = True

    diffs = [STRENGTH_TO_DIFFICULTY[params[c]["strength"]] for c in order]
    difficulty = "hard" if "hard" in diffs else ("mid" if "mid" in diffs else "easy")
    params["_difficulty"] = difficulty
    params["_order"] = order
    return out, order, params


def build_edits(
    real_records: list[dict], num: int = 2000, seed: int = 42, offset: int = 0
) -> list[dict]:
    ensure_dirs()
    AIGC_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    root = pathlib.Path(__file__).resolve().parents[1]
    records = []
    for i in tqdm(range(num), desc="编辑变异合成"):
        src = real_records[(i + offset) % len(real_records)]
        img = imread(root / src["file"])
        if img is None:
            continue
        out_img, labels, params = edit_variation(img, rng)
        rec = save_derived(
            out_img,
            AIGC_DIR,
            f"aigc_{i:06d}",
            "aigc",
            src["origin_id"],
            src.get("scene", "indoor"),
            labels,
            params,
            parent_sha1=src["sha1"],
        )
        rec["aigc_backend"] = "edit"
        records.append(rec)
    return records


# ---------------------------------------------------------------- SD 后端（需 GPU）


def build_sd(real_records: list[dict], num: int = 2000, seed: int = 42, model: str = "runwayml/stable-diffusion-v1-5") -> list[dict]:
    """Stable Diffusion 生成。仅在有 CUDA 时可用；本机无显卡会自动跳过。

    生成时不依赖原图（纯文生图），因此 origin_id 用 sd_xxxxxx 自成一组，
    分组切分时天然不会与真实帧泄漏。
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as e:
        raise RuntimeError(
            "SD 后端需要 diffusers，且需要 NVIDIA 显卡。请先安装：\n"
            "  uv pip install diffusers transformers accelerate\n"
            f"原始错误: {e}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("SD 后端需要 CUDA，当前设备不可用。请改用 --backend edit")

    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    generator = torch.Generator("cuda").manual_seed(seed)

    prompts = [
        "surveillance camera frame, low quality, heavy motion blur, cctv still",
        "security camera still, overexposed highlights, blown out details",
        "night vision camera frame, very dark, near black, cctv",
        "cctv frame with heavy jpeg compression artifacts and blocking",
        "security camera lens covered in dirt and rain streaks",
        "low resolution surveillance image, pixelated and soft",
    ]

    AIGC_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i in tqdm(range(num), desc="SD 生成"):
        prompt = prompts[i % len(prompts)]
        img = pipe(prompt, generator=generator).images[0]
        arr = np.asarray(img)[:, :, ::-1].copy()
        labels = _labels_from_prompt(prompt)
        params = {"_difficulty": "mid", "_order": labels, "prompt": prompt, "model": model}
        rec = save_derived(
            arr, AIGC_DIR, f"sdx_{i:06d}", "aigc", f"sd_{i:06d}",
            "outdoor", labels, params,
        )
        rec["aigc_backend"] = "sd"
        records.append(rec)
    return records


def _labels_from_prompt(prompt: str) -> list[str]:
    p = prompt.lower()
    mapping = [
        ("motion blur", "blur"), ("overexposed", "over_expo"), ("very dark", "dark"),
        ("compression artifacts", "noise"), ("dirt and rain", "occlusion"),
        ("low resolution", "low_res"),
    ]
    return [lab for key, lab in mapping if key in p]


# ---------------------------------------------------------------- CLI


def auto_backend() -> str:
    try:
        import torch

        return "sd" if torch.cuda.is_available() else "edit"
    except Exception:
        return "edit"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="第三数据源：AIGC 生成 / 编辑变异")
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["auto", "edit", "sd"], default="auto")
    ap.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.common import append_manifest, load_manifest

    ensure_dirs()
    real_records = [r for r in load_manifest() if r["source"] == "real"]
    if not real_records:
        print("[aigc] 未找到真实帧，请先运行: uv run python -m src.collect")
        return 1

    backend = args.backend
    if backend == "auto":
        backend = auto_backend()
    print(f"[aigc] 使用后端: {backend}")

    if backend == "sd":
        records = build_sd(real_records, args.num, args.seed, args.model)
    else:
        records = build_edits(real_records, args.num, args.seed)

    append_manifest(records)
    print(f"[aigc] 生成并登记 {len(records)} 条 -> data/aigc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
