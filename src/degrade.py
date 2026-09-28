"""六类程序化降级合成。

六类：low_res(低分辨率) / blur(模糊) / over_expo(过曝) / dark(欠曝黑屏)
      occlusion(遮挡污渍) / noise(压缩噪声)

每类三档强度 mild / moderate / severe。强度档直接决定评测集的难度分层：
    mild -> hard(难)   moderate -> mid(中)   severe -> easy(易)
因为降级越轻微越难判定，越明显越容易判定。这个映射写进 docs/sampling_spec.md。

边界情况：mild 档特意做得接近不可察觉，用来制造"难例"，
这批样本就是标注规范里说的"轻微程度的边界样本"。
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Any, Callable

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    CLASSES,
    PROC_DIR,
    STRENGTH_TO_DIFFICULTY,
    append_manifest,
    assess_quality,
    ensure_dirs,
    imread,
    imwrite,
    rel,
    sha1_of_bytes,
)

STRENGTHS = ["mild", "moderate", "severe"]

# ---------------------------------------------------------------- 单类降级实现


def make_low_res(img: np.ndarray, rng: random.Random, strength: str) -> tuple[np.ndarray, dict]:
    """低分辨率：降采样再放大，保留振铃与细节损失痕迹。"""
    h, w = img.shape[:2]
    ranges = {"mild": (1.5, 2.2), "moderate": (2.5, 3.5), "severe": (4.0, 6.0)}
    scale = rng.uniform(*ranges[strength])
    interp_up = cv2.INTER_CUBIC if strength == "mild" else cv2.INTER_LINEAR
    small = cv2.resize(img, (max(1, int(w / scale)), max(1, int(h / scale))), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=interp_up)
    return out, {"scale": round(scale, 3), "up_interp": int(interp_up)}


def make_blur(img: np.ndarray, rng: random.Random, strength: str) -> tuple[np.ndarray, dict]:
    """模糊：运动模糊 / 高斯失焦 二选一。"""
    ranges = {"mild": (3, 5), "moderate": (7, 11), "severe": (15, 25)}
    lo, hi = ranges[strength]
    k = rng.randrange(lo, hi + 1) | 1  # 保证奇数核

    if rng.random() < 0.5:
        # 运动模糊：随机方向的线核
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        # 随机旋转核，模拟不同运动方向
        angle = rng.uniform(0, 180)
        m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, m, (k, k))
        s = kernel.sum()
        if s > 0:
            kernel /= s
        out = cv2.filter2D(img, -1, kernel)
        kind = "motion"
        params = {"k": k, "kind": kind, "angle": round(angle, 1)}
    else:
        sigma = {  "mild": rng.uniform(0.6, 1.2),
                   "moderate": rng.uniform(1.5, 2.5),
                   "severe": rng.uniform(3.0, 5.0)}[strength]
        out = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
        params = {"k": k, "kind": "gaussian", "sigma": round(sigma, 2)}
    return out, params


def make_over_expo(img: np.ndarray, rng: random.Random, strength: str) -> tuple[np.ndarray, dict]:
    """过曝：gamma<1 提亮，severe 档额外做高光截断。"""
    ranges = {"mild": (0.75, 0.88), "moderate": (0.55, 0.72), "severe": (0.32, 0.5)}
    gamma = rng.uniform(*ranges[strength])
    f = (img.astype(np.float32) / 255.0) ** gamma
    out = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    params: dict[str, Any] = {"gamma": round(gamma, 3)}
    if strength == "severe":
        # 高光溢出：把最亮的 12% 像素直接推到 255，制造细节丢失
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        thr = float(np.percentile(gray, 88))
        mask = gray >= thr
        out[mask] = 255
        params["clip_percentile"] = 88
    return out, params


def make_dark(img: np.ndarray, rng: random.Random, strength: str) -> tuple[np.ndarray, dict]:
    """欠曝 / 黑屏：gamma>1 压暗，severe 档叠加对比度坍缩模拟近纯黑帧。"""
    ranges = {"mild": (1.4, 1.8), "moderate": (1.9, 2.6), "severe": (3.0, 4.5)}
    gamma = rng.uniform(*ranges[strength])
    f = (img.astype(np.float32) / 255.0) ** gamma
    out = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    params: dict[str, Any] = {"gamma": round(gamma, 3)}
    if strength == "severe":
        alpha = rng.uniform(0.25, 0.5)  # 对比度坍缩
        out = cv2.convertScaleAbs(out, alpha=alpha)
        params["contrast_alpha"] = round(alpha, 3)
    return out, params


def make_occlusion(img: np.ndarray, rng: random.Random, strength: str) -> tuple[np.ndarray, dict]:
    """遮挡 / 污渍：随机贴片。四种贴片内容覆盖异物、污点、镜头脏污。"""
    h, w = img.shape[:2]
    spec = {
        "mild": {"count": 1, "size": (0.05, 0.12)},
        "moderate": {"count": rng.randint(1, 2), "size": (0.12, 0.24)},
        "severe": {"count": rng.randint(2, 3), "size": (0.24, 0.42)},
    }[strength]
    out = img.copy()
    patches = []
    for _ in range(int(spec["count"])):
        frac = rng.uniform(*spec["size"])
        side = max(8, int(min(h, w) * frac))
        x = rng.randint(0, max(0, w - side))
        y = rng.randint(0, max(0, h - side))

        mode = rng.choice(["solid", "noise", "dark", "smudge"])
        if mode == "solid":
            patch = np.full((side, side, 3), rng.randint(0, 255), dtype=np.uint8)
        elif mode == "noise":
            patch = np.random.randint(0, 256, (side, side, 3), dtype=np.uint8)
        elif mode == "dark":
            patch = np.full((side, side, 3), rng.randint(0, 45), dtype=np.uint8)
        else:  # smudge：镜头污渍，低频半透明
            base = np.full((side, side, 3), rng.randint(120, 210), dtype=np.uint8)
            patch = cv2.GaussianBlur(base, (0, 0), sigmaX=max(2, side // 8), sigmaY=max(2, side // 8))
            beta = rng.uniform(0.45, 0.75)
            roi = out[y : y + side, x : x + side].astype(np.float32)
            out[y : y + side, x : x + side] = np.clip(
                roi * (1 - beta) + patch.astype(np.float32) * beta, 0, 255
            ).astype(np.uint8)
            patches.append({"mode": mode, "x": x, "y": y, "size": side, "beta": round(beta, 2)})
            continue

        out[y : y + side, x : x + side] = patch
        patches.append({"mode": mode, "x": x, "y": y, "size": side})
    return out, {"patches": patches}


def make_noise(img: np.ndarray, rng: random.Random, strength: str) -> tuple[np.ndarray, dict]:
    """压缩噪声：低码率 JPEG 重编码（块效应 + 蚊式噪声），severe 叠高斯噪声。"""
    ranges = {"mild": (45, 68), "moderate": (20, 42), "severe": (5, 16)}
    q = int(rng.uniform(*ranges[strength]))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img, {"quality": q, "encode_failed": True}
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    params: dict[str, Any] = {"quality": q}
    if strength == "severe":
        sigma = rng.uniform(6, 16)
        g = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + g, 0, 255).astype(np.uint8)
        params["gaussian_sigma"] = round(sigma, 2)
    return out, params


DEGRADE_FUNCS: dict[str, Callable[[np.ndarray, random.Random, str], tuple[np.ndarray, dict]]] = {
    "low_res": make_low_res,
    "blur": make_blur,
    "over_expo": make_over_expo,
    "dark": make_dark,
    "occlusion": make_occlusion,
    "noise": make_noise,
}


# ---------------------------------------------------------------- 组合


def degrade_image(
    img: np.ndarray,
    rng: random.Random,
    classes: list[str] | None = None,
    strengths: dict[str, str] | None = None,
    multi_label_prob: float = 0.25,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """对一张原图施加一组降级，返回 (结果图, 标签列表, 参数字典)。

    multi_label_prob 控制多标签样本比例 —— 真实质检里一帧常同时命中多类。
    """
    if classes is None:
        n = 2 if rng.random() < multi_label_prob else 1
        classes = rng.sample(CLASSES, n)

    # 随机施加顺序：噪声放在最后更接近真实采集链路（先退化后编码）
    order = classes[:]
    rng.shuffle(order)
    if "noise" in order:
        order.remove("noise")
        order.append("noise")

    out = img
    params: dict[str, Any] = {}
    for c in order:
        st = (strengths or {}).get(c) or rng.choice(STRENGTHS)
        out, p = DEGRADE_FUNCS[c](out, rng, st)
        params[c] = {"strength": st, **p}

    # 难度取"最容易被判定的那一类"对应的难度，即最难的那类主导
    # mild->hard 最难，所以整体难度 = 各档中最 mild 的那一档
    diffs = [STRENGTH_TO_DIFFICULTY[params[c]["strength"]] for c in order]
    difficulty = "hard" if "hard" in diffs else ("mid" if "mid" in diffs else "easy")
    params["_difficulty"] = difficulty
    params["_order"] = order
    return out, order, params


def save_derived(
    img: np.ndarray,
    out_dir: pathlib.Path,
    name: str,
    source: str,
    origin_id: str,
    scene: str,
    labels: list[str],
    params: dict[str, Any],
    parent_sha1: str = "",
) -> dict[str, Any]:
    out = out_dir / f"{name}.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"编码失败 {out}")
    data = buf.tobytes()
    out.write_bytes(data)
    q = assess_quality(img)
    return {
        "file": rel(out),
        "sha1": sha1_of_bytes(data),
        "source": source,
        "origin_id": origin_id,     # 继承原图 id —— 分组切分防泄漏的关键
        "labels": labels,
        "params": params,
        "scene": scene,
        "difficulty": params.get("_difficulty", "mid"),
        "width": int(q["width"]),
        "height": int(q["height"]),
        "mean_lum": q["mean_lum"],
        "lap_var": q["lap_var"],
        "parent_sha1": parent_sha1,
        "split": "unassigned",
        "phash": "",
    }


def build_procedural(
    real_records: list[dict],
    num: int = 6000,
    seed: int = 42,
    multi_label_prob: float = 0.25,
) -> list[dict]:
    """从真实清晰帧批量生成程序化降级样本。"""
    ensure_dirs()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict] = []

    if not real_records:
        raise RuntimeError("没有真实帧可供合成，请先运行 collect")

    for i in tqdm(range(num), desc="程序化降级合成"):
        src = real_records[i % len(real_records)]
        img = imread(pathlib.Path(__file__).resolve().parents[1] / src["file"])
        if img is None:
            continue
        out_img, labels, params = degrade_image(img, rng, multi_label_prob=multi_label_prob)
        rec = save_derived(
            out_img,
            PROC_DIR,
            f"proc_{i:06d}",
            "procedural",
            src["origin_id"],
            src.get("scene", "indoor"),
            labels,
            params,
            parent_sha1=src["sha1"],
        )
        records.append(rec)
    return records


# ---------------------------------------------------------------- 预览（D3 肉眼检查）


def write_preview(real_records: list[dict], out_path: pathlib.Path, seed: int = 0) -> pathlib.Path:
    """生成一张对照图：横向六类，纵向三档强度，用于肉眼验收。"""
    rng = random.Random(seed)
    root = pathlib.Path(__file__).resolve().parents[1]
    tile = 160
    src = real_records[0]
    img = imread(root / src["file"])
    img = cv2.resize(img, (tile, tile))

    rows = []
    lab_w = 70
    header = np.full((40, tile * (len(CLASSES) + 1), 3), 255, dtype=np.uint8)
    for j, c in enumerate(CLASSES):
        cv2.putText(header, c, (tile * (j + 1) + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(header, "origin", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    rows.append(np.hstack([np.full((40, lab_w, 3), 255, dtype=np.uint8), header]))

    for st in STRENGTHS:
        row = [img]
        for c in CLASSES:
            out_img, _, _ = degrade_image(img, random.Random(rng.randint(0, 10**6)), classes=[c], strengths={c: st})
            row.append(cv2.resize(out_img, (tile, tile)))
        r = np.hstack(row)
        lab = np.full((r.shape[0], lab_w, 3), 255, dtype=np.uint8)
        cv2.putText(lab, st, (6, tile // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows.append(np.hstack([lab, r]))

    canvas = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="六类程序化降级合成")
    ap.add_argument("--num", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--multi-label-prob", type=float, default=0.25)
    ap.add_argument("--preview", action="store_true", help="只生成对照预览图，不落全量数据")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.common import load_manifest

    ensure_dirs()
    real_records = [r for r in load_manifest() if r["source"] == "real"]
    if not real_records:
        print("[degrade] 未找到真实帧，请先运行: uv run python -m src.collect")
        return 1

    if args.preview:
        p = write_preview(real_records, pathlib.Path("reports/figures/degrade_preview.jpg"))
        print(f"[degrade] 预览图 -> {p}")
        return 0

    records = build_procedural(real_records, args.num, args.seed, args.multi_label_prob)
    append_manifest(records)
    print(f"[degrade] 生成并登记 {len(records)} 条 -> data/procedural")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
