"""预下载 CLIP 权重到仓库内 .cache/clip，供离线使用。

为什么需要这个脚本：
  1. openai/clip-vit-base-patch32 在 HF 上**只有 pytorch_model.bin，没有 safetensors**
     （照着别的仓库写 model.safetensors 会 404）；
  2. transformers 的 HF 缓存偶发把 config.json 截成 0 字节，之后每次加载都报
     "not a valid JSON file"，且缓存命中导致不会重新下载；
  3. .cache/ 不入库（体积大），换机器后需要重新拉取。

用法：
    uv run python scripts/fetch_clip.py
    uv run python scripts/fetch_clip.py --endpoint https://hf-mirror.com

下载完 src.label_vlm.ClipBackend 会自动优先使用本地目录。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request

REPO = "openai/clip-vit-base-patch32"
# 该文件仓库里确实不存在，列出是为了让报错信息有据可循
REQUIRED = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "pytorch_model.bin",
]

MIN_SIZE = {
    "config.json": 1000,
    "pytorch_model.bin": 100 * 1024 * 1024,  # 约 577MB，低于 100MB 一定是没下全
}


def _fetch(base: str, name: str, dest: pathlib.Path) -> bool:
    if dest.exists() and dest.stat().st_size >= MIN_SIZE.get(name, 1):
        print(f"  跳过（已存在 {dest.stat().st_size / 1048576:.1f} MB）：{name}")
        return True
    url = f"{base}/{REPO}/resolve/main/{name}"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  失败：{name} -> {exc}")
        return False
    if len(data) < MIN_SIZE.get(name, 1):
        print(f"  失败：{name} 只有 {len(data)} 字节，疑似重定向页或下载中断")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"  完成：{name}（{len(data) / 1048576:.1f} MB）")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="预下载 CLIP 权重")
    ap.add_argument("--endpoint", default="https://huggingface.co", help="HF 站点，国内可用 https://hf-mirror.com")
    ap.add_argument("--dest", default=None, help="目标目录，默认 <仓库>/.cache/clip")
    args = ap.parse_args()

    dest = pathlib.Path(args.dest) if args.dest else pathlib.Path(__file__).resolve().parents[1] / ".cache" / "clip"
    print(f"目标目录：{dest}\n站点：{args.endpoint}\n")

    ok = all(_fetch(args.endpoint.rstrip("/"), n, dest / n) for n in REQUIRED)
    if not ok:
        print("\n部分文件下载失败。可换镜像重试：")
        print("  uv run python scripts/fetch_clip.py --endpoint https://hf-mirror.com")
        return 1

    total = sum((dest / n).stat().st_size for n in REQUIRED) / 1048576
    print(f"\n全部就绪（{total:.1f} MB）。CLIP 后端将自动使用本地权重。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
