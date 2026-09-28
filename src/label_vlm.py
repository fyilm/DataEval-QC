"""VLM / 人工替代的预标注与标签一致性核对。

四个可插拔后端，按设备与配置自动选择：
    clip       —— open_clip zero-shot，CPU/GPU 都能跑，本机默认
    qwen_local —— Qwen2.5-VL 本地推理，需要 NVIDIA 显卡（租卡后启用）
    qwen_api   —— OpenAI 兼容接口调用 Qwen2.5-VL，需要 API Key
    rule       —— 纯画质指标启发式，零依赖，作为对照基线

闭环设计（这是岗位最看重的部分）：
    规则标签(生成时已知) -> VLM 预标注 -> 两者比对
    -> 不一致样本进入人工复核队列 data/review_queue.csv
    -> 复核量下降比例直接量化（见报告）

注意口径：CLIP 对 low_res / noise 这类低层画质缺陷的判别力天然偏弱，
一致率不高是**预期内的真实现象**，本身就是可写进报告的发现；
不应为了好看而把一致率当 KPI 调参。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from typing import Any, Protocol

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    CLASSES,
    CLASS_CN,
    DATA_DIR,
    MANIFEST_PATH,
    REPORTS_DIR,
    ROOT,
    get_device,
    imread,
    load_manifest,
    write_json,
)

REVIEW_QUEUE = DATA_DIR / "review_queue.csv"


# ---------------------------------------------------------------- 后端协议


class Backend(Protocol):
    name: str

    def predict(self, img: np.ndarray) -> tuple[list[str], dict[str, float]]:
        """返回 (预测标签列表, 各类置信度)。"""


# ---------------------------------------------------------------- rule 后端


class RuleBackend:
    """画质指标启发式。不依赖任何模型，用作对照基线与降级方案。

    阈值必须校准后才能用。初版直接写死经验常数（如 blur 用 lap_var/400），
    与采集端的筛查标准（lap_var >= 80）不自洽，导致大量本已判定为干净的帧
    又被判成模糊 —— 实测干净帧误报率 87%，其中很大一部分是这个原因，
    而不是方法本身不行。所以默认阈值改成由干净帧分布校准得出。
    """

    name = "rule"

    # 指标名 -> (越大越像缺陷?) 由 _indicators 统一计算
    INDICATORS = ("blur_inv", "low_freq_ratio", "bright", "dark_inv", "flat_area", "block_edge")

    def __init__(self, thresholds: "dict[str, float] | None" = None) -> None:
        import cv2

        self.cv2 = cv2
        # 未校准时的兜底值：沿用采集端的筛查标准，至少保证与数据自洽
        self.thresholds = thresholds or {
            "blur_inv": 80.0,      # lap_var 低于此值判 blur（与采集筛查线一致）
            "low_freq_ratio": 0.75,
            "bright": 225.0,       # 与采集端 MEAN_LUM_RANGE 上限一致
            "dark_inv": 45.0,      # 与采集端 MEAN_LUM_RANGE 下限一致
            "flat_area": 0.34,
            "block_edge": 10.0,
        }

    def _indicators(self, img: np.ndarray) -> dict[str, float]:
        """算出各维度的原始指标值（阈值在校准后统一比较）。"""
        cv2 = self.cv2
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        cy, cx = h // 2, w // 2
        low_energy = f[cy - h // 8 : cy + h // 8, cx - w // 8 : cx + w // 8].sum()

        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        return {
            "blur_inv": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "low_freq_ratio": float(low_energy / (f.sum() + 1e-6)),
            "bright": float(gray.mean()),
            "dark_inv": float(gray.mean()),
            "flat_area": float((cv2.Laplacian(sat, cv2.CV_64F).var() < 30).mean()),
            "block_edge": float(np.abs(np.diff(gray.astype(np.int16), axis=0)[:, ::8]).mean()),
        }

    @classmethod
    def calibrate(
        cls,
        clean_images: "list[np.ndarray]",
        p: float = 3.0,
    ) -> "dict[str, float]":
        """用干净帧的指标分布反推阈值与尺度。

        阈值取分位数，尺度取"中位数到阈值的距离"，两者都基于分位数，
        因此对长尾分布稳健 —— 拉普拉斯方差这类指标 mean ± 2*std 会算出
        负阈值（实测 -4439），等于该维度永不触发；换成分位数就没有这个问题。

        p 是唯一旋钮，语义直接：允许约 p% 的干净帧在该指标上被误判。
        偏离阈值达到"中位数到阈值的距离"时软分数约 0.73，刚好过 0.5 线。
        """
        if not clean_images:
            return {}
        base = cls()
        cols: dict[str, list[float]] = {key: [] for key in cls.INDICATORS}
        for img in clean_images:
            if img is None:
                continue
            try:
                ind = base._indicators(img)
            except Exception:
                continue
            for key, v in ind.items():
                cols[key].append(v)

        # 越大越"有缺陷"的指标取上侧，越大越"干净"的指标取下侧
        higher_is_defect = {
            "blur_inv": False,       # 拉普拉斯方差越大越清晰
            "low_freq_ratio": True,
            "bright": True,
            "dark_inv": False,       # 亮度越低越暗
            "flat_area": True,
            "block_edge": True,
        }
        out: dict[str, float] = {}
        for key, vals in cols.items():
            if len(vals) < 2:
                continue
            arr = np.array(vals, dtype=np.float64)
            q50 = float(np.percentile(arr, 50.0))
            if higher_is_defect[key]:
                thr = float(np.percentile(arr, 100.0 - p))
                span = thr - q50
            else:
                thr = float(np.percentile(arr, p))
                span = q50 - thr
            if span <= 1e-9:
                # 分布极度集中的退化情况：给一个与量级相当的兜底尺度
                span = abs(q50) * 0.1 + 1e-6
            out[key] = thr
            out[f"{key}_scale"] = span
        return out

    def predict(self, img: np.ndarray) -> tuple[list[str], dict[str, float]]:
        ind = self._indicators(img)
        th = self.thresholds

        def sig(x: float) -> float:
            return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -30, 30)))))

        def _scale_of(key: str) -> float:
            sd = th.get(f"{key}_scale", 0.0)
            if sd > 0:
                return sd
            # 未校准时的兜底尺度：取阈值的 50%，避免全部判 0 分导致永远输出空标签
            return abs(th.get(key, 0.0)) * 0.5 + 1e-6

        def above(key: str) -> float:
            """指标高于阈值 -> 判该类缺陷，偏离量用标准差归一化。"""
            if key not in th:
                return 0.0
            return sig((ind[key] - th[key]) / _scale_of(key))

        def below(key: str) -> float:
            """指标低于阈值 -> 判该类缺陷。"""
            if key not in th:
                return 0.0
            return sig((th[key] - ind[key]) / _scale_of(key))

        scores = {
            "blur": below("blur_inv"),
            "low_res": above("low_freq_ratio"),
            "over_expo": above("bright"),
            "dark": below("dark_inv"),
            "occlusion": above("flat_area"),
            "noise": above("block_edge"),
        }
        labels = [c for c in CLASSES if scores[c] > 0.5]
        return labels, scores


# ---------------------------------------------------------------- clip 后端


class ClipBackend:
    """CLIP zero-shot（transformers 版）。

    为什么不用 open_clip：open_clip 的 laion 权重仓库在 HF 上已被门控
    （匿名访问返回 Invalid username or password），openai 权重又挂在 timm 的
    HF 仓库上、匿名下载会得到 0 字节文件。transformers 的
    openai/clip-vit-base-patch32 实测可正常匿名下载。

    支持两个权重来源：
      - huggingface.co（默认）
      - hf-mirror.com（HF_ENDPOINT=https://hf-mirror.com 时自动用）
      - 本地文件：CLIP_LOCAL_DIR 指向已下载的模型目录
    """

    MODEL_ID = "openai/clip-vit-base-patch32"
    name = "clip"
    # 仓库内固定存放位置；用 scripts/fetch_clip.py 预下载后完全离线可用
    LOCAL_DIR = ROOT / ".cache" / "clip"

    @staticmethod
    def _as_embedding(out: Any) -> Any:
        """兼容 transformers 4.x / 5.x 的返回类型差异。

        4.x 的 get_text_features / get_image_features 直接返回 tensor；
        5.x 改成返回 BaseModelOutputWithPooling，真正的嵌入在 pooler_output。
        不处理的话 5.x 上报 'BaseModelOutputWithPooling' object has no attribute 'norm'。
        """
        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            return pooled
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    @classmethod
    def _resolve_source(cls) -> str:
        """优先用本地完整权重，其次才走 HF 缓存/联网。

        实测 HF 缓存里的 config.json 有概率被截成 0 字节（下载中断但缓存
        仍判定命中），导致后续一律报 "not a valid JSON file"。本地目录是
        可验证的完整副本，放在最前面可以避免这个坑。
        """
        env = os.environ.get("CLIP_LOCAL_DIR", "")
        if env:
            return env
        need = ("config.json", "preprocessor_config.json", "tokenizer.json", "pytorch_model.bin")
        local = cls.LOCAL_DIR
        if all((local / f).exists() and (local / f).stat().st_size > 0 for f in need):
            return str(local)
        return cls.MODEL_ID

    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        device = get_device()
        self.device = device
        src = self._resolve_source()
        self.src = src
        self.model = CLIPModel.from_pretrained(src).to(str(device)).eval()
        self.processor = CLIPProcessor.from_pretrained(src)

        # 提示词按"画质现象"表述，与画面内容无关，避免 CLIP 偏向物体类别
        self.templates = {
            "low_res": [
                "a low resolution pixelated blurry photo",
                "an image with visible pixelation and blocky artifacts",
            ],
            "blur": ["an out of focus blurry photo", "a motion blurred photo"],
            "over_expo": [
                "an overexposed photo with blown out highlights",
                "a very bright washed out photo",
            ],
            "dark": ["an underexposed very dark photo", "a nearly black dark photo"],
            "occlusion": [
                "a photo with the camera lens partially blocked or covered",
                "a photo with a large obstruction or dirt in front of the camera",
            ],
            "noise": [
                "a photo with heavy jpeg compression artifacts",
                "a grainy noisy heavily compressed photo",
            ],
        }
        # 干净锚点：CLIP zero-shot 判"有没有缺陷"必须先定义"没缺陷"是什么样的。
        # 只给 6 个缺陷提示做 softmax，概率必然被摊到缺陷类上，干净帧会被 100% 误报。
        # 加一个 normal 类后，argmax 落在 normal 即判无缺陷。
        self.templates["normal"] = [
            "a clean sharp high quality photo with good detail",
            "a clear well exposed photo with good lighting and details",
        ]
        self.class_names = list(CLASSES)
        self.all_names = list(CLASSES) + ["normal"]
        self.defect_threshold = 0.18
        self._text_features: "np.ndarray | None" = None
        self._owner: list[str] = []

    def _encode_texts(self) -> tuple["np.ndarray", list[str]]:
        import torch as th

        texts: list[str] = []
        owner: list[str] = []
        for c in self.all_names:
            for tp in self.templates[c]:
                texts.append(tp)
                owner.append(c)
        with th.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(str(self.device))
            emb = self.model.get_text_features(**inputs)
            emb = self._as_embedding(emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy(), owner

    @property
    def text_features(self) -> "np.ndarray":
        if self._text_features is None:
            self._text_features, self._owner = self._encode_texts()
        return self._text_features

    def predict(self, img: "np.ndarray") -> tuple[list[str], dict[str, float]]:
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(str(self.device))
        with self.torch.no_grad():
            feat = self.model.get_image_features(**tensor)
            feat = self._as_embedding(feat)
            feat = (feat / feat.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        sim = feat @ self.text_features.T                      # (1, 2*n_names)
        # 每个类取其两条提示中的最大相似度
        raw = np.array(
            [sim[0, [i for i, o in enumerate(self._owner) if o == c]].max() for c in self.all_names],
            dtype=np.float64,
        )
        # 温度用 CLIP 训练时学到的 logit_scale（约 100），而非随手设的常数
        scale = float(self.model.logit_scale.exp().clamp(max=100.0).item())
        z = raw * scale
        p = np.exp(z - z.max())
        p = p / p.sum()
        pmap = {c: float(v) for c, v in zip(self.all_names, p)}

        # argmax 落在 normal -> 判无缺陷（干净锚点的作用就在这里）
        if pmap["normal"] >= max(pmap[c] for c in self.class_names):
            return [], {c: pmap[c] for c in self.class_names}

        labels = sorted(
            (c for c in self.class_names if pmap[c] > self.defect_threshold),
            key=lambda c: -pmap[c],
        )[:3]
        return labels, {c: pmap[c] for c in self.class_names}


# ---------------------------------------------------------------- qwen 后端


QWEN_PROMPT = (
    "你是视频质检标注员。判断这张监控画面帧存在哪些画质缺陷，"
    f"只能从以下 {len(CLASSES)} 类中选择（可多选，也可都不选）：\n"
    + "\n".join(f"- {k}: {v}" for k, v in CLASS_CN.items())
    + "\n只输出 JSON：{\"labels\": [...]}"
)


class QwenLocalBackend:
    """Qwen2.5-VL 本地推理，需要 NVIDIA 显卡。"""

    name = "qwen_local"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("qwen_local 后端需要 NVIDIA 显卡。本机可用 clip 或 rule 后端。")
        self.device = "cuda"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda"
        )
        self.model_id = model_id

    def predict(self, img: np.ndarray) -> tuple[list[str], dict[str, float]]:
        import base64

        import cv2
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{b64}"},
                {"type": "text", "text": QWEN_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        answer = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return self._parse(answer)

    def _parse(self, answer: str) -> tuple[list[str], dict[str, float]]:
        labels: list[str] = []
        try:
            s = answer[answer.index("{"): answer.rindex("}") + 1]
            labels = [x for x in json.loads(s).get("labels", []) if x in CLASSES]
        except Exception:
            pass
        return labels, {c: (1.0 if c in labels else 0.0) for c in CLASSES}


class QwenApiBackend:
    """OpenAI 兼容接口。需要设置环境变量 QWEN_API_KEY（与 QWEN_BASE_URL）。"""

    name = "qwen_api"

    def __init__(self, model: str = "qwen-vl-max") -> None:
        self.model = model
        self.key = os.environ.get("QWEN_API_KEY", "")
        self.base = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not self.key:
            raise RuntimeError("缺少 QWEN_API_KEY 环境变量。或改用 clip / rule 后端。")

    def predict(self, img: np.ndarray) -> tuple[list[str], dict[str, float]]:
        import base64

        import cv2
        import urllib.request

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode()
        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": QWEN_PROMPT},
                ],
            }],
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode())
        answer = data["choices"][0]["message"]["content"]
        return self._parse(answer)

    def _parse(self, answer: str) -> tuple[list[str], dict[str, float]]:
        labels: list[str] = []
        try:
            s = answer[answer.index("{"): answer.rindex("}") + 1]
            labels = [x for x in json.loads(s).get("labels", []) if x in CLASSES]
        except Exception:
            pass
        return labels, {c: (1.0 if c in labels else 0.0) for c in CLASSES}


# ---------------------------------------------------------------- 调度


def build_backend(name: str = "auto") -> Any:
    if name == "rule":
        return RuleBackend()
    if name == "clip":
        return ClipBackend()
    if name == "qwen_local":
        return QwenLocalBackend()
    if name == "qwen_api":
        return QwenApiBackend()
    if name == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    return QwenLocalBackend()
                except Exception:
                    pass
            return ClipBackend()
        except Exception:
            return RuleBackend()
    raise ValueError(f"未知后端: {name}")


# ---------------------------------------------------------------- 一致性核对


def _iou_multilabel(pred: list[str], gt: list[str]) -> float:
    p, g = set(pred), set(gt)
    if not p and not g:
        return 1.0
    return len(p & g) / len(p | g)


def run_consistency(records: list[dict[str, Any]], backend: Any, limit: int = 0, seed: int = 42) -> dict[str, Any]:
    """跑预标注并与规则标签比对，产出复核队列。"""
    import cv2

    rng = np.random.default_rng(seed)
    if limit and limit < len(records):
        idx = rng.choice(len(records), size=limit, replace=False)
        records = [records[i] for i in sorted(idx)]

    agreement_flags: list[bool] = []
    ious: list[float] = []
    per_class_hit = {c: [0, 0] for c in CLASSES}   # [hit, gt_count]
    review_rows: list[list[str]] = []

    # 失败模式拆解：只报一个一致率说明不了问题，必须区分
    #   - 干净帧被误判有缺陷（false alarm，产线上会浪费人工复检）
    #   - 缺陷帧被判成干净（miss，产线上会漏检，代价更高）
    clean_total = 0
    clean_false_alarm = 0
    defect_total = 0
    defect_missed = 0
    pred_label_counts: list[int] = []
    gt_label_counts: list[int] = []

    for rec in records:
        img = imread(rec["file"])
        if img is None:
            continue
        pred, scores = backend.predict(img)
        gt = rec.get("labels") or []
        ok = set(pred) == set(gt)
        agreement_flags.append(ok)
        ious.append(_iou_multilabel(pred, gt))
        pred_label_counts.append(len(pred))
        gt_label_counts.append(len(gt))
        if gt:
            defect_total += 1
            if not pred:
                defect_missed += 1
        else:
            clean_total += 1
            if pred:
                clean_false_alarm += 1
        for c in CLASSES:
            if c in gt:
                per_class_hit[c][1] += 1
                if c in pred:
                    per_class_hit[c][0] += 1
        if not ok:
            review_rows.append([
                rec["file"], rec.get("origin_id", ""), rec.get("source", ""),
                "|".join(gt), "|".join(pred),
                "|".join(f"{c}:{scores.get(c, 0):.2f}" for c in CLASSES),
                "pending",
            ])

    n = len(agreement_flags)
    exact = float(np.mean(agreement_flags)) if n else 0.0
    mean_iou = float(np.mean(ious)) if n else 0.0

    REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_QUEUE.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "origin_id", "source", "rule_labels", "vlm_labels", "scores", "status"])
        w.writerows(review_rows)

    report = {
        "backend": backend.name,
        "sampled": n,
        "exact_match_rate": round(exact, 4),
        "mean_iou": round(mean_iou, 4),
        "per_class_recall": {c: round(v[0] / v[1], 4) if v[1] else None for c, v in per_class_hit.items()},
        "review_queue_size": len(review_rows),
        "review_reduction": round(1 - len(review_rows) / n, 4) if n else 0.0,
        "failure_modes": {
            "clean_total": clean_total,
            "clean_false_alarm": clean_false_alarm,
            "clean_false_alarm_rate": round(clean_false_alarm / clean_total, 4) if clean_total else None,
            "defect_total": defect_total,
            "defect_missed": defect_missed,
            "defect_missed_rate": round(defect_missed / defect_total, 4) if defect_total else None,
            "mean_pred_labels": round(float(np.mean(pred_label_counts)), 3) if pred_label_counts else 0.0,
            "mean_gt_labels": round(float(np.mean(gt_label_counts)), 3) if gt_label_counts else 0.0,
        },
        "note": (
            "一致率=精确匹配；IoU=多标签重合度；复核量下降比例=1-不一致数/抽样数。"
            "failure_modes 区分两类失败：干净帧误报(clean_false_alarm_rate)与缺陷帧漏检(defect_missed_rate)，"
            "后者在质检场景下代价更高。"
        ),
    }
    return report


def _fmt_pct(v: "float | None") -> str:
    return "n/a" if v is None else f"{v:.2%}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VLM 预标注与一致性核对")
    ap.add_argument("--backend", choices=["auto", "clip", "qwen_local", "qwen_api", "rule"], default="auto")
    ap.add_argument("--limit", type=int, default=0, help="只抽样 N 张核对，0 表示全量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="vlm_consistency_report.json", help="报告文件名（便于多后端对比并存）")
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="用干净真实帧的指标分布校准 rule 后端阈值（仅对 rule 生效）",
    )
    args = ap.parse_args(argv)

    records = load_manifest()
    active = [r for r in records if not r.get("dedup_removed", False)]
    if not active:
        print("[label_vlm] manifest 为空")
        return 1

    backend = build_backend(args.backend)
    print(f"[label_vlm] 后端: {backend.name}")

    if args.calibrate and isinstance(backend, RuleBackend):
        clean_recs = [r for r in active if not (r.get("labels") or [])][:300]
        clean_imgs = [imread(r["file"]) for r in clean_recs]
        clean_imgs = [x for x in clean_imgs if x is not None]
        th = RuleBackend.calibrate(clean_imgs)
        if th:
            backend.thresholds.update(th)
            print(
                f"[label_vlm] 已用 {len(clean_imgs)} 张干净帧校准阈值: "
                + ", ".join(f"{kk}={v:.2f}" for kk, v in th.items() if not kk.endswith("_scale"))
            )

    report = run_consistency(active, backend, args.limit, args.seed)
    if args.calibrate and isinstance(backend, RuleBackend):
        report["thresholds"] = {k: round(v, 4) for k, v in backend.thresholds.items()}

    out_path = REPORTS_DIR / args.out
    write_json(report, out_path)
    fm = report["failure_modes"]
    print(
        f"[label_vlm] 精确一致率 {report['exact_match_rate']:.2%} | 多标签 IoU {report['mean_iou']:.3f}\n"
        f"            抽样 {report['sampled']} 张，复核队列 {report['review_queue_size']} 张"
        f"（复核量下降 {report['review_reduction']:.2%}）\n"
        f"            干净帧误报率 {_fmt_pct(fm['clean_false_alarm_rate'])}"
        f"（{fm['clean_false_alarm']}/{fm['clean_total']}） | "
        f"缺陷帧漏检率 {_fmt_pct(fm['defect_missed_rate'])}"
        f"（{fm['defect_missed']}/{fm['defect_total']}）\n"
        f"            平均标签数 预测 {fm['mean_pred_labels']} vs 真实 {fm['mean_gt_labels']}\n"
        f"            报告 -> {out_path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
