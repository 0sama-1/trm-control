#!/usr/bin/env python3
"""Mechanistic analysis: *why* does the recursive trunk underperform?

Reward-only comparisons (analyze_phase1/analyze_phase2) establish *that* the
recursive trunk does not beat a memoryless MLP. This script mines the
per-episode training telemetry already stored in every result file
(episode_history: return, avg100_return, actor_loss, critic_loss) to
discriminate between competing explanations:

  E1 "optimisation/conditioning deficit" -- the recursive trunk is a worse
     function approximator *to train* in this setting (most of its recursive
     computation receives no gradient: with H_cycles=3 only the final outer
     cycle is backpropped, so 14 of 21 shared-cell applications per forward
     pass are untrained). Prediction: systematically *higher critic loss* at
     matched episode budget, i.e. a worse value fit, not merely worse reward.

  E2 "slower learning, same asymptote" -- recursion costs sample-efficiency
     but eventually converges to parity. Prediction: TRM curve lags the DNN
     curve early but converges by the end of the budget.

  E3 "lower asymptote" -- recursion caps achievable performance.
     Prediction: persistent end-of-budget gap.

  E4 "instability" -- recursion increases across-seed variance.
     Prediction: larger seed spread in final return.

Outputs per (env, condition, model) cell:
  - learning curve of avg100_return, averaged across seeds
  - critic-loss and actor-loss trajectories (median across seeds)
  - early (first 25% of budget) vs late (final 25%) critic loss
  - normalised area under the learning curve (sample-efficiency proxy)
  - episodes-to-threshold (first episode reaching 90% of the per-env best
    final avg100_return achieved by any architecture)
  - across-seed std of final avg100_return

Stdlib only (plus matplotlib/numpy for the figures, via make_figures_mechanism).
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def condition_key(r: dict) -> str:
    if r.get("delay") == "none":
        return "none"
    return "fixed-aug" if r.get("augment_obs") else "fixed-noaug"


def load(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        d.pop("step_history", None)  # large, not needed here
        rows.append(d)
    return rows


def series(hist: list[dict], key: str) -> list[float]:
    out = []
    for e in hist:
        v = e.get(key)
        out.append(float("nan") if v is None else float(v))
    return out


def nanmedian(vals: list[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return statistics.median(clean) if clean else float("nan")


def mean_curve(curves: list[list[float]]) -> list[float]:
    """Element-wise mean across seeds, truncated to the shortest run."""
    if not curves:
        return []
    n = min(len(c) for c in curves)
    out = []
    for i in range(n):
        vals = [c[i] for c in curves if not math.isnan(c[i])]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def window_stat(vals: list[float], lo_frac: float, hi_frac: float) -> float:
    n = len(vals)
    if n == 0:
        return float("nan")
    lo, hi = int(n * lo_frac), max(int(n * hi_frac), int(n * lo_frac) + 1)
    return nanmedian(vals[lo:hi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE / "analysis_mechanism.json", type=Path)
    ap.add_argument("--curves-out", default=HERE / "mechanism_curves.json", type=Path)
    args = ap.parse_args()

    rows = load(PROJECT / "results_v3") + load(PROJECT / "results_phase2")
    print(f"Loaded {len(rows)} runs")

    # group: (env, condition, model) -> list of runs
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["env_id"], condition_key(r), r["model"])].append(r)

    # per-env best final performance, for the episodes-to-threshold metric
    best_final: dict[str, float] = {}
    for (env, _cond, _m), runs in cells.items():
        for r in runs:
            v = r["final_avg100_return"]
            best_final[env] = max(best_final.get(env, -math.inf), v)

    summary: dict[str, dict] = {}
    curves: dict[str, dict] = {}

    for (env, cond, model), runs in sorted(cells.items()):
        key = f"{env}__{cond}__{model}"
        ret_curves, critic_curves, actor_curves = [], [], []
        finals, to_thresh = [], []

        target = best_final[env]
        # threshold: 90% of the way from a pessimistic floor (worst final seen
        # for this env) to the per-env best final. Reward scales differ per
        # env and are negative, so anchor on the observed range rather than 0.
        worst_final = min(
            r["final_avg100_return"] for (e, _c, _m), rs in cells.items() if e == env for r in rs
        )
        span = target - worst_final
        thresh = worst_final + 0.9 * span if span > 0 else target

        for r in runs:
            hist = r.get("episode_history") or []
            ret = series(hist, "avg100_return")
            ret_curves.append(ret)
            critic_curves.append(series(hist, "critic_loss"))
            actor_curves.append(series(hist, "actor_loss"))
            finals.append(r["final_avg100_return"])
            hit = next((i + 1 for i, v in enumerate(ret) if not math.isnan(v) and v >= thresh), None)
            to_thresh.append(hit)

        ret_mean = mean_curve(ret_curves)
        critic_mean = mean_curve(critic_curves)
        actor_mean = mean_curve(actor_curves)

        # normalised AUC of the learning curve: mean over episodes, rescaled so
        # 0 = worst final for the env, 1 = best final for the env. Higher =
        # reached good performance sooner and held it.
        auc = float("nan")
        if ret_mean and span > 0:
            clean = [v for v in ret_mean if not math.isnan(v)]
            if clean:
                auc = (sum(clean) / len(clean) - worst_final) / span

        hits = [t for t in to_thresh if t is not None]
        summary[key] = {
            "env": env,
            "condition": cond,
            "model": model,
            "n_seeds": len(runs),
            "param_count": runs[0].get("param_count"),
            "wall_clock_median_s": statistics.median([r["wall_clock_seconds"] for r in runs]),
            "final_return_median": statistics.median(finals),
            "final_return_std": statistics.stdev(finals) if len(finals) > 1 else 0.0,
            "critic_loss_early": window_stat(critic_mean, 0.0, 0.25),
            "critic_loss_late": window_stat(critic_mean, 0.75, 1.0),
            "actor_loss_early": window_stat(actor_mean, 0.0, 0.25),
            "actor_loss_late": window_stat(actor_mean, 0.75, 1.0),
            "learning_curve_auc_norm": auc,
            "episodes_to_90pct": statistics.median(hits) if hits else None,
            "n_seeds_reaching_90pct": len(hits),
            "threshold_used": thresh,
        }
        curves[key] = {
            "avg100_return": ret_mean,
            "critic_loss": critic_mean,
            "actor_loss": actor_mean,
        }

    # ---- headline contrasts: TRM vs DNN within each (env, condition) ----
    contrasts = {}
    for (env, cond, model), _runs in sorted(cells.items()):
        if model != "trm":
            continue
        dnn_key = f"{env}__{cond}__dnn"
        trm_key = f"{env}__{cond}__trm"
        if dnn_key not in summary:
            continue
        d, t = summary[dnn_key], summary[trm_key]
        def ratio(a, b):
            if b in (None, 0) or (isinstance(b, float) and math.isnan(b)) or \
               a is None or (isinstance(a, float) and math.isnan(a)):
                return None
            return a / b
        contrasts[f"{env}__{cond}"] = {
            "critic_loss_late_trm_over_dnn": ratio(t["critic_loss_late"], d["critic_loss_late"]),
            "critic_loss_early_trm_over_dnn": ratio(t["critic_loss_early"], d["critic_loss_early"]),
            "auc_trm_minus_dnn": (t["learning_curve_auc_norm"] - d["learning_curve_auc_norm"])
                if not (math.isnan(t["learning_curve_auc_norm"]) or math.isnan(d["learning_curve_auc_norm"])) else None,
            "final_return_trm_minus_dnn": t["final_return_median"] - d["final_return_median"],
            "seed_std_trm_over_dnn": ratio(t["final_return_std"], d["final_return_std"]),
            "episodes_to_90pct_trm": t["episodes_to_90pct"],
            "episodes_to_90pct_dnn": d["episodes_to_90pct"],
            "wallclock_trm_over_dnn": ratio(t["wall_clock_median_s"], d["wall_clock_median_s"]),
        }

    # ---- sign tests: is the value-fit (critic loss) deficit systematic? ----
    # For each candidate trunk, count the (env, condition) cells in which its
    # critic loss exceeds the DNN's at matched episode budget. The Transformer
    # is the key control: comparable parameter count (~797k vs ~788k), NO
    # recursion and NO gradient truncation. If the deficit is specific to the
    # recursive trunks and absent for the Transformer, it implicates the
    # recursion/truncation mechanism rather than model size.
    def binom_ge(k: int, n: int) -> float:
        return sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n) if n else 1.0

    env_conds = sorted({(s["env"], s["condition"]) for s in summary.values()})
    sign_tests = {}
    for model in ("trm", "trm_hier6", "trm_singlez", "transformer", "lstm"):
        for phase, field in (("early", "critic_loss_early"), ("late", "critic_loss_late")):
            worse, better, deltas = 0, 0, []
            for env, cond in env_conds:
                mk, dk = f"{env}__{cond}__{model}", f"{env}__{cond}__dnn"
                if mk not in summary or dk not in summary:
                    continue
                a, b = summary[mk][field], summary[dk][field]
                if any(isinstance(v, float) and math.isnan(v) for v in (a, b)) or b == 0:
                    continue
                deltas.append(a / b)
                worse += a > b
                better += a < b
            n = worse + better
            sign_tests[f"{model}__critic_{phase}_vs_dnn"] = {
                "n_cells": n,
                "n_cells_worse_than_dnn": worse,
                "ratios": [round(x, 3) for x in deltas],
                "median_ratio": round(statistics.median(deltas), 3) if deltas else None,
                "one_sided_p_worse": round(binom_ge(worse, n), 5) if n else None,
            }

    out = {
        "n_runs": len(rows),
        "cells": summary,
        "trm_vs_dnn_contrasts": contrasts,
        "critic_loss_sign_tests": sign_tests,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(args.curves_out, "w") as f:
        json.dump(curves, f, default=str)
    print(f"Wrote {args.out} and {args.curves_out}")

    print("\n=== TRM vs DNN mechanistic contrasts ===")
    for k, v in contrasts.items():
        print(f"\n{k}")
        for kk, vv in v.items():
            if isinstance(vv, float):
                print(f"   {kk:38s} {vv:+.3f}")
            else:
                print(f"   {kk:38s} {vv}")


if __name__ == "__main__":
    main()
