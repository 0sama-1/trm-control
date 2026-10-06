#!/usr/bin/env python3
"""Generate the report's figures from the v3 result set and the pre-registered
analysis (analysis_v3_final.json). Stdlib + matplotlib only.

Figures produced (paper/figures/):
  fig1_reward_by_arch_env.pdf/png  -- grouped bar chart, reward per architecture x env
  fig2_wallclock_by_arch_env.pdf/png -- grouped bar chart, wall-clock (log scale)
  fig3_trm_advantage_trend.pdf/png -- TRM-advantage-vs-complexity trend (vs DNN, vs Transformer)
  fig4_param_count.pdf/png -- parameter count per architecture
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIG_DIR = PROJECT / "phase1" / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

analysis = json.load(open(HERE / "analysis_v3_final.json"))
cells = analysis["cells"]

ARCHS = ["dnn", "transformer", "trm", "lstm"]
ARCH_LABELS = {"dnn": "DNN", "transformer": "Transformer", "trm": "TRM", "lstm": "LSTM"}
ENVS = ["Pendulum-v1", "MountainCarContinuous-v0"]
ENV_LABELS = {"Pendulum-v1": "Pendulum-v1\n(low complexity)", "MountainCarContinuous-v0": "MountainCarContinuous-v0\n(higher complexity)"}
COLORS = {"dnn": "#4C72B0", "transformer": "#DD8452", "trm": "#55A868", "lstm": "#C44E52"}

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def cell(arch, env):
    return cells.get(f"{arch}__{env}", {})


# --- Figure 1: reward by architecture x env ---
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
for ax, env in zip(axes, ENVS):
    vals = [cell(a, env).get("reward_median", np.nan) for a in ARCHS]
    iqrs = [cell(a, env).get("reward_iqr", [np.nan, np.nan]) for a in ARCHS]
    lo = [v - iqr[0] for v, iqr in zip(vals, iqrs)]
    hi = [iqr[1] - v for v, iqr in zip(vals, iqrs)]
    x = np.arange(len(ARCHS))
    bars = ax.bar(x, vals, yerr=[lo, hi], capsize=4,
                  color=[COLORS[a] for a in ARCHS])
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
    ax.set_title(ENV_LABELS[env])
    ax.set_ylabel("Final reward (median, IQR whiskers)")
    ax.axhline(0, color="black", linewidth=0.5)
fig.suptitle("Final reward by architecture and environment (n=6 per cell: 3 seeds x 2 delay conditions)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_reward_by_arch_env.pdf")
fig.savefig(FIG_DIR / "fig1_reward_by_arch_env.png")
plt.close(fig)

# --- Figure 2: wall-clock by architecture x env (log scale) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.35
x = np.arange(len(ARCHS))
for i, env in enumerate(ENVS):
    vals = [cell(a, env).get("wall_clock_median_s", np.nan) / 60.0 for a in ARCHS]  # minutes
    ax.bar(x + (i - 0.5) * width, vals, width, label=ENV_LABELS[env].split("\n")[0],
           color=[COLORS[a] for a in ARCHS], alpha=[1.0, 0.55][i])
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
ax.set_ylabel("Wall-clock training time (minutes, log scale)")
ax.set_title("Wall-clock training time by architecture and environment (median)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_wallclock_by_arch_env.pdf")
fig.savefig(FIG_DIR / "fig2_wallclock_by_arch_env.png")
plt.close(fig)

# --- Figure 3: TRM-advantage vs complexity trend (vs DNN, vs Transformer only -- LSTM is secondary) ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
rung_labels = ["Pendulum-v1", "MountainCarContinuous-v0"]
x = [0, 1]
for baseline, marker, color in [("dnn", "o", "#4C72B0"), ("transformer", "s", "#DD8452")]:
    trend = analysis["trend_tests"][f"trm_advantage_vs_{baseline}"]
    y = trend["values_by_rung"]
    ax.plot(x, y, marker=marker, color=color, label=f"TRM − {ARCH_LABELS[baseline]}", linewidth=2, markersize=8)
sign_p = {
    "dnn__Pendulum-v1": analysis["sign_tests"]["trm_vs_dnn__Pendulum-v1"]["one_sided_p_majority_sign"],
    "dnn__MountainCarContinuous-v0": analysis["sign_tests"]["trm_vs_dnn__MountainCarContinuous-v0"]["one_sided_p_majority_sign"],
    "transformer__Pendulum-v1": analysis["sign_tests"]["trm_vs_transformer__Pendulum-v1"]["one_sided_p_majority_sign"],
    "transformer__MountainCarContinuous-v0": analysis["sign_tests"]["trm_vs_transformer__MountainCarContinuous-v0"]["one_sided_p_majority_sign"],
}
ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(rung_labels)
ax.set_ylabel("Median reward delta (TRM − baseline)")
ax.set_title("TRM's reward advantage vs. task complexity\n(pre-registered primary comparison)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_trm_advantage_trend.pdf")
fig.savefig(FIG_DIR / "fig3_trm_advantage_trend.png")
plt.close(fig)

# --- Figure 4: parameter count per architecture ---
fig, ax = plt.subplots(figsize=(6, 4))
pcounts = [cell(a, "Pendulum-v1").get("param_count", np.nan) for a in ARCHS]
ax.bar(range(len(ARCHS)), pcounts, color=[COLORS[a] for a in ARCHS])
ax.set_xticks(range(len(ARCHS)))
ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
ax.set_ylabel("Parameter count (actor + critic)")
ax.set_title("Model size by trunk architecture")
for i, v in enumerate(pcounts):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_param_count.pdf")
fig.savefig(FIG_DIR / "fig4_param_count.png")
plt.close(fig)

print("Figures written to", FIG_DIR)
for f in sorted(FIG_DIR.glob("*.png")):
    print(" -", f.name)

# --- Emit a small table-data JSON the report text can quote verbatim ---
table = {"cells": cells, "sign_p": sign_p}
json.dump(table, open(HERE / "figure_table_data.json", "w"), indent=2)
