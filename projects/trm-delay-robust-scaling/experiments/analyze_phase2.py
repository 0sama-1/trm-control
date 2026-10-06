#!/usr/bin/env python3
"""Phase 2 analysis: 6 architectures x 3 conditions on BipedalWalker-v3.

Extends analyze_phase1.py's approach for the expanded design:
  - TRM-variant ablation: trm (base), trm_hier6, trm_singlez
  - Augmentation ablation: fixed-delay with augment_obs on vs. off
  - Same pooled sign test / Page's trend test machinery, adapted to a
    single-environment, 3-condition ladder instead of a 2-environment one.

Reads every phase2/results/*.json (one per run: env_id, arch, model, delay,
delay_value, augment_obs, seed, final_avg100_return, wall_clock_seconds,
param_count -- plus episode_history/step_history, not used here), and reports:

  1. Per (model, condition) cell: median/IQR reward, median wall-clock,
     param count.
  2. TRM-variant comparison: trm vs trm_hier6, trm vs trm_singlez (does the
     specific recursive structure matter, independent of "TRM vs baseline"?)
  3. Augmentation ablation: for each architecture, fixed-aug vs fixed-noaug
     paired sign test (does the augmentation fix's benefit hold/grow at
     this complexity, or does removing it hurt every architecture equally?)
  4. Primary TRM-vs-baseline comparison at the fixed-aug condition, for
     continuity with Phase 1's decision rule (pooled across seeds only,
     n=3, since there is a single environment and a single delay condition
     in this comparison -- weaker power than Phase 1, stated explicitly).

Stdlib only.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
            d.pop("episode_history", None)
            d.pop("step_history", None)  # keep this script lightweight
            rows.append(d)
    return rows


def binom_sign_test_p(n_success: int, n_total: int) -> float:
    if n_total == 0:
        return 1.0
    total = 0.0
    for k in range(n_success, n_total + 1):
        total += math.comb(n_total, k) * (0.5 ** n_total)
    return total


def condition_key(r: dict) -> str:
    if r["delay"] == "none":
        return "none"
    return "fixed-aug" if r.get("augment_obs") else "fixed-noaug"


def cell_stats(rows: list[dict], model: str, cond: str) -> dict:
    vals = [r["final_avg100_return"] for r in rows if r["model"] == model and condition_key(r) == cond]
    wct = [r["wall_clock_seconds"] for r in rows if r["model"] == model and condition_key(r) == cond]
    params = [r["param_count"] for r in rows if r["model"] == model and condition_key(r) == cond]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "reward_median": statistics.median(vals),
        "reward_iqr": (sorted(vals)[len(vals) // 4], sorted(vals)[(3 * len(vals)) // 4]) if len(vals) >= 4 else (min(vals), max(vals)),
        "wall_clock_median_s": statistics.median(wct) if wct else None,
        "param_count": params[0] if params else None,
    }


def paired_sign_test(rows: list[dict], model_a: str, model_b: str, cond: str) -> dict:
    by_seed = {}
    for r in rows:
        if condition_key(r) != cond or r["model"] not in (model_a, model_b):
            continue
        by_seed.setdefault(r["seed"], {})[r["model"]] = r["final_avg100_return"]
    deltas = [p[model_a] - p[model_b] for p in by_seed.values() if model_a in p and model_b in p]
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_eff = n_pos + n_neg
    p = binom_sign_test_p(max(n_pos, n_neg), n_eff) if n_eff else 1.0
    return {"model_a": model_a, "model_b": model_b, "condition": cond,
            "n_pairs": len(deltas), "n_pos": n_pos, "n_neg": n_neg, "deltas": deltas,
            "one_sided_p_majority_sign": p}


def augmentation_ablation(rows: list[dict], model: str) -> dict:
    """fixed-aug vs fixed-noaug, paired by seed -- does augmentation help
    this architecture at BipedalWalker's complexity?"""
    by_seed = {}
    for r in rows:
        if r["model"] != model or r["delay"] == "none":
            continue
        key = "aug" if r.get("augment_obs") else "noaug"
        by_seed.setdefault(r["seed"], {})[key] = r["final_avg100_return"]
    deltas = [p["aug"] - p["noaug"] for p in by_seed.values() if "aug" in p and "noaug" in p]
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_eff = n_pos + n_neg
    p = binom_sign_test_p(max(n_pos, n_neg), n_eff) if n_eff else 1.0
    return {"model": model, "n_pairs": len(deltas), "n_pos_aug_better": n_pos,
            "n_neg_aug_worse": n_neg, "deltas": deltas, "one_sided_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="phase2/results", type=Path)
    ap.add_argument("--out", default="analysis_phase2_final.json", type=Path)
    args = ap.parse_args()

    rows = load_results(args.results_dir)
    print(f"Loaded {len(rows)} result files from {args.results_dir}")
    if not rows:
        print("No results yet.")
        return

    models = sorted(set(r["model"] for r in rows))
    conditions = ["none", "fixed-aug", "fixed-noaug"]
    print(f"Models: {models}")

    summary = {"n_results": len(rows), "models": models, "cells": {}, "trm_variant_tests": {},
               "augmentation_ablation": {}, "primary_trm_vs_baseline": {}}

    for model in models:
        for cond in conditions:
            summary["cells"][f"{model}__{cond}"] = cell_stats(rows, model, cond)

    if "trm" in models:
        for variant in ("trm_hier6", "trm_singlez"):
            if variant in models:
                for cond in conditions:
                    summary["trm_variant_tests"][f"trm_vs_{variant}__{cond}"] = paired_sign_test(rows, "trm", variant, cond)

    for model in models:
        summary["augmentation_ablation"][model] = augmentation_ablation(rows, model)

    for baseline in ("dnn", "transformer", "lstm"):
        if "trm" in models and baseline in models:
            summary["primary_trm_vs_baseline"][f"trm_vs_{baseline}"] = paired_sign_test(rows, "trm", baseline, "fixed-aug")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
