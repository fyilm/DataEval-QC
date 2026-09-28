"""真实清晰帧采集。

默认数据源：HuggingFace 上的 COCO parquet 镜像（detection-datasets/coco）。
选它的原因：cocodataset.org 官方域名在本机超时，HuggingFace 实测 16MB/s；
COCO 来源清晰（CC BY 4.0）、分辨率够（约 640x480），适合当作"真实清晰帧"。

也支持从本地视频/图片目录抽帧（--from-video / --from-dir），无需 ffmpeg，
走 OpenCV 的 VideoCapture。

关键：采集阶段会做质量筛查，只保留"清晰"帧入库（"真实清晰帧"这一源必须真的是清晰的），
被筛掉的天然低质帧不丢，单独落到 reports/rejected_real.jsonl 备查。
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import urllib.request
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    DATA_DIR,
    REAL_DIR,
    REPORTS_DIR,
    append_manifest,
    assess_quality,
    ensure_dirs,
    imread,
    rel,
    sha1_of_bytes,
)

HF_REPO = "detection-datasets/coco"
HF_API_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/data"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / ".cache"

# 质量筛查阈值（写入 docs/sampling_spec.md，改这里必须同步改文档）
MIN_SIDE = 320          # 短边下限，太小的图无法承载后续降级
MAX_LONG_SIDE = 960     # 长边上限，超过则等比缩放，控制磁盘占用
MIN_LAPLACIAN_VAR = 80.0  # 拉普拉斯方差下限：低于此值视为天然模糊，不能算"清晰帧"
MEAN_LUM_RANGE = (45.0, 225.0)  # 平均亮度合法区间：过暗/过曝的不算"清晰帧"


def _http_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def list_shards() -> list[dict[str, Any]]:
    """列出 HuggingFace 上该数据集的 parquet 分片。"""
    items = _http_json(HF_API_TREE)
    shards = [it for it in items if it["path"].endswith(".parquet")]
    shards.sort(key=lambda x: x["path"])
    return shards


def download_shard(path: str, dest_dir: pathlib.Path = CACHE_DIR) -> pathlib.Path:
    """带缓存的分片下载，避免重复拉 485MB。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pathlib.Path(path).name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[collect] 命中缓存 {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest

    url = f"{HF_RESOLVE}/{path}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"下载 {dest.name}"
        ) as pbar:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(dest)
    return dest


def _find_image_column(pf) -> str:
    """在 parquet schema 里找到存图像字节的列（不同数据集叫法不一致）。"""
    for field in pf.schema_arrow:
        if field.type is None:
            continue
        t = str(field.type).lower()
        if "struct" in t and "bytes" in t:
            return field.name
    # 退而求其次：找名字里带 image 的列
    for field in pf.schema_arrow:
        if "image" in field.name.lower():
            return field.name
    raise RuntimeError(f"未找到图像列，schema={pf.schema_arrow}")


def iter_parquet_images(parquet_path: pathlib.Path, batch_size: int = 64):
    """逐批读出图像字节，避免一次性把 485MB 全读进内存。"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    col = _find_image_column(pf)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[col]):
        coldata = batch.column(0)
        for item in coldata.to_pylist():
            if not item:
                continue
            if isinstance(item, dict):
                data = item.get("bytes") or item.get("content")
                if data is None:
                    # 嵌套一层的情况
                    for v in item.values():
                        if isinstance(v, bytes):
                            data = v
                            break
                if data is not None:
                    yield data


def _decode_bgr(data: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        arr = np.asarray(img)[:, :, ::-1]  # RGB -> BGR
        return np.ascontiguousarray(arr)
    except Exception:
        return None


def resize_keep_aspect(img: np.ndarray, max_long: int = MAX_LONG_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_long:
        return img
    s = max_long / m
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


# 夜间专项：夜视画面整体偏暗属正常曝光，不是缺陷。
# 判定 dark 的标准是"细节不可辨"，不是"绝对亮度低"——这条写进 annotation_spec.md。
NIGHT_LUM_RANGE = (18.0, 88.0)
NIGHT_MIN_LAPLACIAN_VAR = 60.0


def is_clean_frame(img: np.ndarray, night: bool = False) -> tuple[bool, str]:
    """判断是否能作为"真实清晰帧"入库。返回 (是否合格, 不合格原因)。

    night=True 走夜间口径：亮度区间放宽到 NIGHT_LUM_RANGE，清晰度阈值略降，
    因为夜间画面天然噪点更多、细节对比更低。
    """
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        return False, f"分辨率过小 {w}x{h}"
    q = assess_quality(img)
    if q["lap_var"] < (NIGHT_MIN_LAPLACIAN_VAR if night else MIN_LAPLACIAN_VAR):
        return False, f"天然模糊 lap_var={q['lap_var']:.1f}"
    lo, hi = NIGHT_LUM_RANGE if night else MEAN_LUM_RANGE
    if not (lo <= q["mean_lum"] <= hi):
        return False, f"亮度越界 mean_lum={q['mean_lum']:.1f}"
    return True, ""


def classify_scene(img: np.ndarray) -> str:
    """规则化场景分类：室内 / 室外 / 夜视。

    阈值口径固定在此函数内，并在 docs/sampling_spec.md 记录；
    一旦改动必须同步文档并给评测集增加新的 changelog 条目。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_lum = float(gray.mean())

    # 夜视：整体亮度低；或亮度偏低且带蓝紫色调（夜间监控常见）
    if mean_lum < 60.0:
        return "night"
    if mean_lum < 95.0:
        b, g, r = (float(img[:, :, i].mean()) for i in range(3))
        if b > r + 12 and b > g:
            return "night"

    # 天空：上 1/3 区域里明亮且低饱和的像素占比
    top = img[: max(1, h // 3)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    sky_ratio = float(((val > 150) & (sat < 70)).mean())

    # 植被：绿通道明显高于红通道
    green_ratio = float((img[:, :, 1].astype(np.int16) - img[:, :, 2].astype(np.int16) > 20).mean())

    outdoor_score = sky_ratio + green_ratio
    return "outdoor" if outdoor_score > 0.18 else "indoor"


def save_real_image(img: np.ndarray, idx: int, scene: str | None = None) -> dict[str, Any]:
    """落盘一张真实帧并返回 manifest 记录。"""
    origin_id = f"real_{idx:06d}"
    out = REAL_DIR / f"{origin_id}.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"JPEG 编码失败: {out}")
    out.write_bytes(buf.tobytes())

    q = assess_quality(img)
    return {
        "file": rel(out),
        "sha1": sha1_of_bytes(buf.tobytes()),
        "source": "real",
        "origin_id": origin_id,
        "labels": [],                       # 真实清晰帧：无缺陷
        "params": {},
        "scene": scene or classify_scene(img),
        "difficulty": "none",               # 清晰帧不参与难度分层
        "width": int(q["width"]),
        "height": int(q["height"]),
        "mean_lum": q["mean_lum"],
        "lap_var": q["lap_var"],
        "split": "unassigned",
        "phash": "",
    }


def collect_from_coco(
    num: int = 1500, max_shards: int = 2, skip_shards: int = 0, night_quota: int = 0
) -> list[dict]:
    """从 COCO parquet 采集。

    num         —— 常规清晰帧配额
    night_quota —— 夜间帧专项配额。COCO 里夜景天然稀少（实测 1500 张里仅 55 张），
                   不做专项采集的话"夜视"这一层撑不起 6类×3场景×3难度 的分层配额。
    """
    shards = list_shards()
    if not shards:
        raise RuntimeError("HuggingFace 上未找到 parquet 分片")

    records: list[dict] = []
    rejected: list[dict] = []
    idx = 0
    seen = 0
    n_normal = 0
    n_night = 0

    for shard in shards[skip_shards : skip_shards + max_shards]:
        if n_normal >= num and n_night >= night_quota:
            break
        local = download_shard(shard["path"])
        print(f"[collect] 扫描分片 {shard['path']} ...")
        for data in iter_parquet_images(local):
            if n_normal >= num and n_night >= night_quota:
                break
            seen += 1
            img = _decode_bgr(data)
            if img is None:
                continue
            img = resize_keep_aspect(img)

            night_ok, night_why = is_clean_frame(img, night=True)
            if n_night < night_quota and night_ok:
                idx += 1
                n_night += 1
                records.append(save_real_image(img, idx, scene="night"))
                continue

            ok, why = is_clean_frame(img)
            if n_normal < num and ok:
                idx += 1
                n_normal += 1
                records.append(save_real_image(img, idx))
                continue

            if len(rejected) < 5000:
                rejected.append({"reason": why if not ok else night_why, "seen": seen})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "rejected_real.jsonl").open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[collect] 扫描 {seen} 张，入库 {len(records)} 张清晰帧"
        f"（常规 {n_normal} + 夜间 {n_night}），筛除 {len(rejected)} 张"
        f"（明细见 reports/rejected_real.jsonl）"
    )
    return records


def collect_from_dir(src: pathlib.Path, num: int = 1500) -> list[dict]:
    """从本地图片目录采集。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    records = []
    idx = 0
    for p in tqdm(files, desc="扫描本地图片"):
        if len(records) >= num:
            break
        img = imread(p)
        if img is None:
            continue
        img = resize_keep_aspect(img)
        ok, _ = is_clean_frame(img)
        if not ok:
            continue
        idx += 1
        records.append(save_real_image(img, idx))
    return records


def collect_from_video(video: pathlib.Path, num: int = 1500, every: int = 30) -> list[dict]:
    """从本地视频抽帧（无需 ffmpeg，走 OpenCV VideoCapture）。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    records = []
    idx = 0
    fid = 0
    with tqdm(desc="抽帧") as pbar:
        while len(records) < num:
            ret, frame = cap.read()
            if not ret:
                break
            if fid % every == 0:
                frame = resize_keep_aspect(frame)
                ok, _ = is_clean_frame(frame)
                if ok:
                    idx += 1
                    records.append(save_real_image(frame, idx))
                    pbar.update(1)
            fid += 1
    cap.release()
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="采集真实清晰帧")
    ap.add_argument("--num", type=int, default=1500)
    ap.add_argument("--from-dir", type=str, default=None, help="本地图片目录")
    ap.add_argument("--from-video", type=str, default=None, help="本地视频文件")
    ap.add_argument("--max-shards", type=int, default=2)
    ap.add_argument("--skip-shards", type=int, default=0)
    ap.add_argument("--night-quota", type=int, default=300, help="夜间帧专项配额，撑起夜视分层")
    args = ap.parse_args(argv)

    ensure_dirs()
    if args.from_dir:
        records = collect_from_dir(pathlib.Path(args.from_dir), args.num)
    elif args.from_video:
        records = collect_from_video(pathlib.Path(args.from_video), args.num)
    else:
        records = collect_from_coco(args.num, args.max_shards, args.skip_shards, args.night_quota)

    append_manifest(records)
    print(f"[collect] 登记 {len(records)} 条 -> data/manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
