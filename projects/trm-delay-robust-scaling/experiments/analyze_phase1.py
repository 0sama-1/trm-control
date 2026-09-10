#!/usr/bin/env python3
"""Phase 1 analysis: pooled sign test + Page's trend test on the
TRM-advantage delta, per the pre-registered decision rule in
ideas/proposal-final.json (Method section, post-fix).

Reads every results/*.json written by src/trainer.py::_write_results_summary
(one per run: env_id, arch, model, delay, delay_value, seed,
final_avg100_return, wall_clock_seconds, param_count), aggregates per
(model, env_id, delay) cell across seeds, and reports:

  - median/IQR reward, wall-clock, and param count per cell
  - the pooled one-sided sign test (n = seeds x delay conditions per
    architecture-pair per environment) on sign(reward_TRM - reward_baseline)
  - Page's L trend statistic across the 2-rung environment ladder
    (Pendulum-v1 -> MountainCarContinuous-v0), as a corroborating
    monotonicity check (secondary, not the sole criterion)

Stdlib only (no scipy dependency, to avoid another Katana install).
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

COMPLEXITY_ORDER = ["Pendulum-v1", "MountainCarContinuous-v0"]  # rung 0, rung 1


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def binom_sign_test_p(n_success: int, n_total: int) -> float:
    """One-sided exact binomial p-value for >= n_success successes out of
    n_total trials under p=0.5 (the sign-test null)."""
    if n_total == 0:
        return 1.0
    total = 0.0
    for k in range(n_success, n_total + 1):
        total += math.comb(n_total, k) * (0.5 ** n_total)
    return total


def pooled_sign_test(rows: list[dict], model_a: str, model_b: str, env_id: str) -> dict:
    """Pool the sign of (reward_a - reward_b) across seeds x delay conditions
    for one environment. Returns n, n_favoring_a, p-value (one-sided, H1: a>b)."""
    by_key = defaultdict(dict)
    for r in rows:
        if r["env_id"] != env_id or r["model"] not in (model_a, model_b):
            continue
        key = (r["delay"], r["seed"])
        by_key[key][r["model"]] = r["final_avg100_return"]

    deltas = []
    for key, pair in by_key.items():
        if model_a in pair and model_b in pair:
            deltas.append(pair[model_a] - pair[model_b])

    n = len(deltas)
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_effective = n_pos + n_neg  # ties dropped, standard sign-test convention
    p = binom_sign_test_p(max(n_pos, n_neg), n_effective) if n_effective > 0 else 1.0
    return {
        "model_a": model_a, "model_b": model_b, "env_id": env_id,
        "n_pairs": n, "n_pos": n_pos, "n_neg": n_neg, "n_ties": n - n_effective,
        "deltas": deltas, "one_sided_p_majority_sign": p,
    }


def pages_trend_L(cell_medians_by_rung: list[float]) -> dict:
    """Page's L trend statistic across ordinal rungs (here: 2 rungs, so this
    reduces to a simple directional check -- reported for completeness and
    to keep the analysis pipeline uniform if a 3rd rung (Phase 2) is added
    later). ranks: 1..k for k rungs, L = sum(rank_i * value_rank_i)."""
    k = len(cell_medians_by_rung)
    ranks = sorted(range(k), key=lambda i: cell_medians_by_rung[i])
    rank_of = {i: r + 1 for r, i in enumerate(ranks)}
    L = sum((i + 1) * rank_of[i] for i in range(k))
    monotonic_increasing = all(
        cell_medians_by_rung[i] <= cell_medians_by_rung[i + 1]
        for i in range(k - 1)
    )
    return {"L": L, "k": k, "monotonic_increasing": monotonic_increasing,
            "values_by_rung": cell_medians_by_rung}


def cell_stats(rows: list[dict], model: str, env_id: str, delay: str | None = None) -> dict:
    vals = [r["final_avg100_return"] for r in rows
            if r["model"] == model and r["env_id"] == env_id
            and (delay is None or r["delay"] == delay)]
    wct = [r["wall_clock_seconds"] for r in rows
           if r["model"] == model and r["env_id"] == env_id
           and (delay is None or r["delay"] == delay)]
    params = [r["param_count"] for r in rows
              if r["model"] == model and r["env_id"] == env_id
              and (delay is None or r["delay"] == delay)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "reward_median": statistics.median(vals),
        "reward_iqr": (sorted(vals)[len(vals) // 4], sorted(vals)[(3 * len(vals)) // 4]) if len(vals) >= 4 else (min(vals), max(vals)),
        "wall_clock_median_s": statistics.median(wct) if wct else None,
        "param_count": params[0] if params else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", type=Path)
    ap.add_argument("--out", default="analysis_summary.json", type=Path)
    args = ap.parse_args()

    rows = load_results(args.results_dir)
    print(f"Loaded {len(rows)} result files from {args.results_dir}")

    models = sorted(set(r["model"] for r in rows))
    envs = sorted(set(r["env_id"] for r in rows))
    print(f"Models: {models}  Envs: {envs}")

    summary = {"n_results": len(rows), "models": models, "envs": envs, "cells": {}, "sign_tests": {}, "trend_tests": {}}

    for model in models:
        for env in envs:
            summary["cells"][f"{model}__{env}"] = cell_stats(rows, model, env)

    for baseline in ("dnn", "transformer", "lstm"):
        if "trm" not in models or baseline not in models:
            continue
        for env in envs:
            key = f"trm_vs_{baseline}__{env}"
            summary["sign_tests"][key] = pooled_sign_test(rows, "trm", baseline, env)

    for baseline in ("dnn", "transformer", "lstm"):
        if "trm" not in models or baseline not in models:
            continue
        medians = []
        for env in COMPLEXITY_ORDER:
            if env not in envs:
                medians = None
                break
            trm_med = cell_stats(rows, "trm", env).get("reward_median")
            base_med = cell_stats(rows, baseline, env).get("reward_median")
            if trm_med is None or base_med is None:
                medians = None
                break
            medians.append(trm_med - base_med)
        if medians:
            summary["trend_tests"][f"trm_advantage_vs_{baseline}"] = pages_trend_L(medians)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
