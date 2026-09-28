"""实验入口：读 yaml 配置跑消融网格，结果逐条落盘 jsonl。

消融口径（这是任务书里没写死、必须先定死的定义）：
    r = 训练集中合成缺陷帧的占比（procedural + aigc 合计）
    真实侧固定为 train split 里全部真实清晰帧（895 张，无法再扩充）

    r = 0     -> 真实 895，合成 0       （无正样本，退化对照组）
    r = 0.25  -> 真实 895，合成 298
    r = 0.50  -> 真实 895，合成 895
    r = 0.75  -> 真实 895，合成 2685
    r = 1.0   -> 真实 0，  合成 3580    （无负样本，退化对照组）

为什么这么定义：任务书把 r 直接当"总样本量固定下的占比"，但本项目真实帧只有
1500 张，r=0 时根本凑不出 5600 张真实帧。所以真实侧取全部可用真实帧、
合成侧按 r 反推数量。r=0 和 r=1 是两个退化对照点，
曲线在中间出现峰值 —— 这正是"合成数据先升后降"结论的来源。

诚实声明：该设计下训练总量随 r 变化，曲线左端同时包含"数据量不足"的效应。
报告里会明确标注这一点，不把它包装成纯"分布偏移"结论。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    CONFIGS_DIR,
    REPORTS_DIR,
    ROOT,
    get_device,
    load_manifest,
    set_seed,
    suggest_num_workers,
    write_json,
)
from src.dataset import QCDataset  # noqa: E402
from src.evaluate import evaluate_logits, search_thresholds, slice_metrics  # noqa: E402
from src.train import MODELS, benchmark_throughput, build_model, collect_logits, train_one  # noqa: E402

RESULTS_PATH = REPORTS_DIR / "experiment_results.jsonl"


# ---------------------------------------------------------------- 数据配比


def make_train_mix(
    train_records: list[dict[str, Any]], ratio: float, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按合成占比 r 构造训练集。返回 (记录列表, 配比说明)。"""
    real = [r for r in train_records if r["source"] == "real"]
    synth = [r for r in train_records if r["source"] in ("procedural", "aigc")]
    rng = np.random.default_rng(seed)

    if ratio >= 1.0:
        k = int(len(real) * 4)      # r=1.0：不含真实帧，合成量取 4 倍真实量以保持量级可比
        use_real: list[dict] = []
    else:
        k = int(round(len(real) * ratio / max(1e-9, (1 - ratio))))
        use_real = list(real)

    k = min(k, len(synth))
    idx = rng.choice(len(synth), size=k, replace=False)
    use_synth = [synth[i] for i in sorted(idx)]

    mix = use_real + use_synth
    n = max(1, len(mix))
    info = {
        "ratio_target": ratio,
        "n_real": len(use_real),
        "n_synthetic": len(use_synth),
        "ratio_actual": round(len(use_synth) / n, 4),
        "n_total": len(mix),
    }
    return mix, info


# ---------------------------------------------------------------- 单次实验


def run_single(cfg: dict[str, Any], seed: int, out_name: str) -> dict[str, Any]:
    device = get_device(cfg.get("device"))
    records = [r for r in load_manifest() if not r.get("dedup_removed", False)]
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    test_recs = [r for r in records if r["split"] == "test"]

    eval_recs = _load_eval_records(records)

    mix, mix_info = make_train_mix(train_recs, cfg["ratio"], seed)
    img_size = cfg.get("img_size", 160)
    epochs = cfg.get("epochs", 10)
    batch_size = cfg.get("batch_size", 64)
    model_name = cfg["model"]

    train_ds = QCDataset(mix, img_size, train=True, seed=seed)
    val_ds = QCDataset(val_recs, img_size, train=False, seed=seed)
    test_ds = QCDataset(test_recs, img_size, train=False, seed=seed)
    eval_ds = QCDataset(eval_recs, img_size, train=False, seed=seed)

    from torch.utils.data import DataLoader

    nw = suggest_num_workers(device)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    dl_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    dl_eval = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    t0 = time.perf_counter()
    ckpt_dir = ROOT / "checkpoints" / out_name
    res = train_one(
        model_name, train_ds, val_ds, device,
        epochs=epochs, batch_size=batch_size, lr=cfg.get("lr", 1e-3),
        img_size=img_size, seed=seed, out_dir=ckpt_dir,
        log_every=cfg.get("log_every", 0),
    )
    train_seconds = round(time.perf_counter() - t0, 1)

    model = build_model(model_name, pretrained=False).to(device)
    ck = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])

    vl, vt = collect_logits(model, dl_val, device)
    tl, tt = collect_logits(model, dl_test, device)
    el, et = collect_logits(model, dl_eval, device)

    # 阈值：召回约束下最大化精确率，只在 val 上搜
    thr, thr_detail = search_thresholds(vl, vt, recall_floor=cfg.get("recall_floor", 0.95))

    eval_metrics = evaluate_logits(el, et, thr)
    test_metrics = evaluate_logits(tl, tt, thr)
    eval_metrics["threshold_0p5"] = evaluate_logits(el, et, 0.5)

    np.savez_compressed(
        ckpt_dir / "logits.npz",
        val_logits=vl, val_targets=vt, test_logits=tl, test_targets=tt,
        eval_logits=el, eval_targets=et, thresholds=thr,
    )

    slices = slice_metrics(eval_recs, el, et, thr)

    out = {
        "exp_id": out_name,
        "model": model_name,
        "ratio": cfg["ratio"],
        "seed": seed,
        "img_size": img_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "mix": mix_info,
        "train_seconds": train_seconds,
        "best_val_macro_f1": res["best_val_macro_f1"],
        "eval": eval_metrics,
        "eval_threshold_0p5": eval_metrics["threshold_0p5"],
        "test": test_metrics,
        "slices": slices,
        "threshold_detail": thr_detail,
    }
    return out


def _load_eval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """读冻结评测集 eval_v1.csv 对应的记录。"""
    import csv

    path = ROOT / "data" / "eval" / "eval_v1.jsonl"
    if not path.exists():
        return [r for r in records if r["split"] == "eval"]
    files = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                files.add(json.loads(line)["file"])
    out = [r for r in records if r["file"] in files]
    if not out:
        out = [r for r in records if r["split"] == "eval"]
    return out


# ---------------------------------------------------------------- benchmark


def cmd_benchmark(cfg: dict[str, Any]) -> int:
    device = get_device(cfg.get("device"))
    img_size = cfg.get("img_size", 160)
    batch = cfg.get("batch_size", 64)
    print(f"[benchmark] device={device} img_size={img_size} batch={batch}")
    rows = []
    for m in cfg.get("models", list(MODELS)):
        model = build_model(m, pretrained=False).to(device)
        th = benchmark_throughput(model, device, img_size, batch)
        rows.append({"model": m, **th})
        print(f"    {m:<22} {th['img_per_sec']:>8.1f} img/s")
    # 一个 epoch 6400 张、10 epoch、30 组 的耗时外推
    print("\n    外推（假设 6400 张/epoch、10 epochs、每组 3 seed）:")
    for r in rows:
        sec_ep = 6400 / r["img_per_sec"]
        per_run = sec_ep * 10 * 3
        print(
            f"    {r['model']:<22} 单组 3seed ≈ {per_run/3600:.2f} h   "
            f"5档×3seed ≈ {per_run*5/3600:.2f} h   "
            f"2模型×5档×3seed ≈ {per_run*10/3600:.2f} h"
        )
    write_json({"device": str(device), "rows": rows}, REPORTS_DIR / "benchmark.json")
    return 0


# ---------------------------------------------------------------- grid


def cmd_grid(cfg: dict[str, Any]) -> int:
    seeds = cfg.get("seeds", [42, 43, 44])
    models = cfg.get("models", list(MODELS))
    ratios = cfg.get("ratios", [0.0, 0.25, 0.5, 0.75, 1.0])

    combos = [(m, r) for m in models for r in ratios]
    total = len(combos) * len(seeds)
    print(f"[grid] 共 {total} 次训练（{len(models)} 模型 × {len(ratios)} 占比 × {len(seeds)} seed）")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：已成功写入结果的组合直接跳过。
    # CPU 上全量要十几小时，中断一次就得从头再来代价太大；
    # GPU 上同理，遇到某组 OOM 也能修完接着跑。
    existing: set[str] = set()
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if "error" not in d and d.get("exp_id"):
                    existing.add(d["exp_id"])
    if existing:
        print(f"[grid] 检测到 {len(existing)} 组已有结果，将跳过")

    done = 0
    skipped = 0
    for m, r in combos:
        for s in seeds:
            exp_id = f"{m}_r{int(round(r*100)):03d}_s{s}"
            if exp_id in existing:
                skipped += 1
                done += 1
                print(f"  [skip {done}/{total}] {exp_id} 已有结果")
                continue
            cfg_i = dict(cfg)
            cfg_i["model"] = m
            cfg_i["ratio"] = r
            try:
                res = run_single(cfg_i, s, exp_id)
            except Exception as e:
                print(f"  [FAIL] {exp_id}: {e}")
                res = {"exp_id": exp_id, "model": m, "ratio": r, "seed": s, "error": str(e)}
            with RESULTS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            done += 1
            if "error" not in res:
                print(
                    f"  [{done}/{total}] {exp_id}  eval_macroF1={res['eval']['macro_f1']:.4f}  "
                    f"ECE={res['eval']['ece_multilabel']:.4f}  ({res['train_seconds']}s)"
                )
            else:
                print(f"  [{done}/{total}] {exp_id} 失败")
    print(f"[grid] 完成（新跑 {done - skipped} 组，跳过 {skipped} 组），结果 -> {RESULTS_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="实验入口")
    ap.add_argument("--config", default=str(CONFIGS_DIR / "ablation.yaml"))
    ap.add_argument("--mode", choices=["grid", "benchmark", "single"], default="grid")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "benchmark":
        return cmd_benchmark(cfg)
    if args.mode == "single":
        cfg["model"] = args.model or cfg.get("models", [MODELS[0]])[0]
        cfg["ratio"] = args.ratio if args.ratio is not None else cfg.get("ratios", [0.5])[0]
        res = run_single(cfg, args.seed, f"single_{cfg['model']}_r{int(cfg['ratio']*100)}_s{args.seed}")
        write_json(res, REPORTS_DIR / "single_exp.json")
        print(
            f"[single] eval macroF1={res['eval']['macro_f1']:.4f} ECE={res['eval']['ece_multilabel']:.4f}"
        )
        return 0
    return cmd_grid(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
