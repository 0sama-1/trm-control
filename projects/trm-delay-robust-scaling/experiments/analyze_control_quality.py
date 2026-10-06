#!/usr/bin/env python3
"""Control-quality re-analysis of the Phase 1 + Phase 2 runs (zero new compute).

The Phase 1/2 report scores trunks on final reward, wall-clock and parameter
count. For a control/robotics reading those are incomplete: a controller is
also judged on how the *command signal* behaves (actuator wear, energy) and on
what it costs per decision in real time. Two such quantities are already
recoverable from the banked runs at no cost:

  1. Command regularity. Every run stores `step_history` -- the per-step
     action actually emitted by the policy -- so mean |a_t - a_{t-1}| ("action
     rate", the chatter proxy) and mean |a_t| are directly computable, and
     with them the scale-free ratio rate/|a|.

  2. Reward at matched wall-clock. Every run stores `episode_history`
     (avg100 return per episode) and total `wall_clock_seconds`. Reading the
     slower trunk's curve at the *fraction* of its run corresponding to the
     baseline's total time answers "what does each trunk buy you per hour?"
     rather than "per episode", which is the practitioner's question.

CAVEAT, stated up front and carried into the plan: the stored actions are the
*exploration-noisy* ones (DDPG adds OU noise inside `_select_action`), so the
rate statistic mixes policy behaviour with a common noise process, and a
degenerate policy (constant output) scores as maximally smooth. This script
therefore establishes a *candidate* signal and its confounds -- the
deterministic, matched-return measurement is Phase 3 workstream W0.

Usage (from the repo root):
    python3 projects/trm-delay-robust-scaling/experiments/analyze_control_quality.py

Writes: projects/trm-delay-robust-scaling/experiments/analysis_control_quality.json
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULT_DIRS = [PROJECT / "phase1" / "results", PROJECT / "phase2" / "results"]
OUT = HERE / "analysis_control_quality.json"

# Last N episodes of each run define the "trained policy" window.
WINDOW_EPISODES = 50
# MountainCarContinuous is excluded from the pooled tests: the median run never
# reaches the goal (report S6.2), so its action trace is a near-constant
# policy and its "smoothness" is passivity, not control quality.
POOLED_ENVS = ("Pendulum-v1", "BipedalWalker-v3")


def sign_test(k: int, n: int) -> float:
    """One-sided P(X >= k) under Binomial(n, 1/2)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def condition(run: dict) -> str:
    if run["delay"] == "none":
        return "none"
    return "fixed-aug" if run["augment_obs"] else "fixed-noaug"


def load_runs() -> list[dict]:
    runs = []
    for d in RESULT_DIRS:
        for f in sorted(glob.glob(str(d / "*.json"))):
            j = json.load(open(f))
            steps = j.get("step_history") or []
            if not steps:
                continue
            last_ep = max(r["episode"] for r in steps)
            cut = last_ep - WINDOW_EPISODES
            acts = np.asarray(
                [r["action"] for r in steps if r["episode"] > cut], dtype=np.float32
            )
            if acts.ndim == 1:
                acts = acts[:, None]
            rate = float(np.abs(np.diff(acts, axis=0)).mean())
            mag = float(np.abs(acts).mean())
            runs.append(
                dict(
                    env=j["env_id"],
                    cond=condition(j),
                    model=j["model"],
                    seed=j["seed"],
                    final_return=j["final_avg100_return"],
                    wall_clock_s=j["wall_clock_seconds"],
                    param_count=j["param_count"],
                    action_rate=rate,
                    action_magnitude=mag,
                    action_rate_norm=rate / mag if mag else float("nan"),
                    saturation_frac=float((np.abs(acts) > 0.99).mean()),
                    episode_history=j["episode_history"],
                )
            )
    return runs


def regularity_tests(runs: list[dict]) -> dict:
    """Per-seed paired comparison of every trunk against the DNN baseline."""
    index = {(r["env"], r["cond"], r["seed"], r["model"]): r for r in runs}
    out = {}
    for metric in ("action_rate", "action_rate_norm"):
        for model in ("trm", "trm_hier6", "trm_singlez", "transformer", "lstm"):
            ratios, wins = [], 0
            for (env, cond, seed, m), r in index.items():
                if m != model or env not in POOLED_ENVS:
                    continue
                base = index.get((env, cond, seed, "dnn"))
                if base is None:
                    continue
                ratios.append(r[metric] / base[metric])
                wins += int(r[metric] < base[metric])
            if ratios:
                out[f"{model}__{metric}_vs_dnn"] = dict(
                    n_pairs=len(ratios),
                    n_smoother_than_dnn=wins,
                    median_ratio=round(float(np.median(ratios)), 3),
                    one_sided_p_smoother=round(sign_test(wins, len(ratios)), 5),
                )
    return out


def matched_wallclock(runs: list[dict]) -> dict:
    """What each trunk has reached when the DNN baseline's run is already over.

    Assumes time per episode is roughly constant within a run (true here: fixed
    episode budget, fixed max_steps, no curriculum), so the DNN's total time
    corresponds to the fraction `T_dnn / T_x` of trunk x's episode sequence.
    """
    by_cell = defaultdict(dict)
    for r in runs:
        by_cell[(r["env"], r["cond"], r["seed"])][r["model"]] = r

    rows, deltas_by_env = [], defaultdict(list)
    for key in sorted(by_cell):
        cell = by_cell[key]
        if "dnn" not in cell:
            continue
        dnn = cell["dnn"]
        for model, r in cell.items():
            if model == "dnn":
                continue
            frac = min(1.0, dnn["wall_clock_s"] / r["wall_clock_s"])
            hist = r["episode_history"]
            idx = max(0, int(frac * len(hist)) - 1)
            at_t = hist[idx]["avg100_return"]
            rows.append(
                dict(
                    env=key[0], cond=key[1], seed=key[2], model=model,
                    dnn_wallclock_min=round(dnn["wall_clock_s"] / 60, 1),
                    dnn_final=round(dnn["final_return"], 1),
                    model_final=round(r["final_return"], 1),
                    model_at_dnn_wallclock=round(at_t, 1),
                    model_episodes_done_at_dnn_wallclock=hist[idx]["episode"],
                )
            )
            deltas_by_env[(key[0], model)].append(at_t - dnn["final_return"])

    summary = {
        f"{env}__{model}": dict(
            n=len(v),
            median_delta_vs_dnn_final=round(float(np.median(v)), 1),
        )
        for (env, model), v in deltas_by_env.items()
    }
    return dict(rows=rows, summary=summary)


def cell_table(runs: list[dict]) -> dict:
    cells = defaultdict(list)
    for r in runs:
        cells[(r["env"], r["cond"], r["model"])].append(r)
    return {
        "__".join(k): dict(
            n=len(v),
            return_median=round(float(np.median([x["final_return"] for x in v])), 1),
            action_rate_median=round(float(np.median([x["action_rate"] for x in v])), 4),
            action_magnitude_median=round(float(np.median([x["action_magnitude"] for x in v])), 3),
            action_rate_norm_median=round(float(np.median([x["action_rate_norm"] for x in v])), 3),
            saturation_median=round(float(np.median([x["saturation_frac"] for x in v])), 3),
            param_count=v[0]["param_count"],
        )
        for k, v in sorted(cells.items())
    }


def main() -> None:
    runs = load_runs()
    payload = dict(
        n_runs=len(runs),
        window_episodes=WINDOW_EPISODES,
        pooled_envs=list(POOLED_ENVS),
        caveats=[
            "Actions are exploration-noisy (OU noise added in _select_action); "
            "the rate statistic mixes policy behaviour with a common noise process.",
            "A degenerate constant policy scores as maximally smooth -- regularity "
            "is only interpretable jointly with return.",
            "MountainCarContinuous-v0 excluded from pooled tests (floor effect).",
            "Matched-wall-clock reading assumes constant time per episode within a run.",
        ],
        cells=cell_table(runs),
        regularity_sign_tests=regularity_tests(runs),
        matched_wallclock=matched_wallclock(runs),
    )
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB), {len(runs)} runs")

    print("\nCommand regularity vs DNN (lower action rate = smoother):")
    for k, v in payload["regularity_sign_tests"].items():
        if "norm" in k:
            continue
        print(f"  {k:38s} {v['n_smoother_than_dnn']}/{v['n_pairs']}  "
              f"median ratio {v['median_ratio']:.2f}  p={v['one_sided_p_smoother']:.4f}")
    print("\nReward at matched wall-clock (trunk reading - DNN final):")
    for k, v in sorted(payload["matched_wallclock"]["summary"].items()):
        print(f"  {k:44s} n={v['n']:2d}  median delta {v['median_delta_vs_dnn_final']:+.1f}")


if __name__ == "__main__":
    main()
