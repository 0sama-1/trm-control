#!/usr/bin/env python3
"""Generate Phase 2 report figures from analysis_phase2_final.json. Stdlib + matplotlib only.

Figures produced (phase2/report/figures/):
  fig1_reward_by_arch_cond.pdf/png   -- grouped bar chart, reward per architecture x condition
  fig2_wallclock_by_arch_cond.pdf/png -- grouped bar chart, wall-clock (log scale)
  fig3_trm_overhead.pdf/png          -- TRM-family wall-clock overhead vs DNN, fixed-aug condition
  fig4_augmentation_ablation.pdf/png -- fixed-aug vs fixed-noaug reward delta per architecture
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIG_DIR = PROJECT / "phase2" / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

analysis = json.load(open(HERE / "analysis_phase2_final.json"))
cells = analysis["cells"]

ARCHS = ["dnn", "transformer", "lstm", "trm", "trm_hier6", "trm_singlez"]
ARCH_LABELS = {"dnn": "DNN", "transformer": "Transformer", "lstm": "LSTM",
               "trm": "TRM", "trm_hier6": "TRM-hier6", "trm_singlez": "TRM-singlez"}
CONDS = ["none", "fixed-aug", "fixed-noaug"]
COND_LABELS = {"none": "no-delay", "fixed-aug": "delay+aug", "fixed-noaug": "delay, no-aug"}
COLORS = {"dnn": "#4C72B0", "transformer": "#DD8452", "lstm": "#C44E52",
          "trm": "#55A868", "trm_hier6": "#8172B2", "trm_singlez": "#937860"}

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def cell(arch, cond):
    return cells.get(f"{arch}__{cond}", {})


# --- Figure 1: reward by architecture x condition ---
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.25
x = np.arange(len(ARCHS))
for i, cond in enumerate(CONDS):
    vals = [cell(a, cond).get("reward_median", np.nan) for a in ARCHS]
    iqrs = [cell(a, cond).get("reward_iqr", [np.nan, np.nan]) for a in ARCHS]
    lo = [v - iqr[0] for v, iqr in zip(vals, iqrs)]
    hi = [iqr[1] - v for v, iqr in zip(vals, iqrs)]
    ax.bar(x + (i - 1) * width, vals, width, yerr=[lo, hi], capsize=3,
           label=COND_LABELS[cond], color=[COLORS[a] for a in ARCHS],
           alpha=[1.0, 0.65, 0.4][i])
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
ax.set_ylabel("Final reward (median, IQR whiskers)")
ax.set_title("BipedalWalker-v3: final reward by architecture and condition (n=3 seeds)")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_reward_by_arch_cond.pdf")
fig.savefig(FIG_DIR / "fig1_reward_by_arch_cond.png")
plt.close(fig)

# --- Figure 2: wall-clock by architecture x condition (log scale) ---
fig, ax = plt.subplots(figsize=(10, 5))
for i, cond in enumerate(CONDS):
    vals = [cell(a, cond).get("wall_clock_median_s", np.nan) / 60.0 for a in ARCHS]
    ax.bar(x + (i - 1) * width, vals, width, label=COND_LABELS[cond],
           color=[COLORS[a] for a in ARCHS], alpha=[1.0, 0.65, 0.4][i])
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
ax.set_ylabel("Wall-clock training time (minutes, log scale)")
ax.set_title("BipedalWalker-v3: wall-clock training time by architecture and condition (median)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_wallclock_by_arch_cond.pdf")
fig.savefig(FIG_DIR / "fig2_wallclock_by_arch_cond.png")
plt.close(fig)

# --- Figure 3: TRM-family wall-clock overhead vs DNN, fixed-aug condition ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
dnn_wc = cell("dnn", "fixed-aug").get("wall_clock_median_s")
overhead_archs = ["transformer", "lstm", "trm", "trm_hier6", "trm_singlez"]
overheads = [cell(a, "fixed-aug").get("wall_clock_median_s", np.nan) / dnn_wc for a in overhead_archs]
bars = ax.bar(range(len(overhead_archs)), overheads, color=[COLORS[a] for a in overhead_archs])
ax.set_xticks(range(len(overhead_archs)))
ax.set_xticklabels([ARCH_LABELS[a] for a in overhead_archs], rotation=20, ha="right")
ax.set_ylabel("Wall-clock overhead vs. DNN (x)")
ax.set_title("BipedalWalker-v3, delay+aug condition:\nwall-clock overhead relative to DNN")
ax.axhline(1, color="black", linewidth=0.7, linestyle="--")
for i, v in enumerate(overheads):
    ax.text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_trm_overhead.pdf")
fig.savefig(FIG_DIR / "fig3_trm_overhead.png")
plt.close(fig)

# --- Figure 4: augmentation ablation (fixed-aug - fixed-noaug reward delta) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ablation = analysis["augmentation_ablation"]
deltas_by_arch = {a: ablation[a]["deltas"] for a in ARCHS if a in ablation}
x = np.arange(len(ARCHS))
medians = [float(np.median(deltas_by_arch[a])) for a in ARCHS]
for i, a in enumerate(ARCHS):
    pts = deltas_by_arch[a]
    ax.scatter([i] * len(pts), pts, color=COLORS[a], zorder=3, s=30)
ax.bar(x, medians, color=[COLORS[a] for a in ARCHS], alpha=0.35, zorder=2)
ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
ax.set_ylabel("Reward delta (aug − no-aug), per seed + median bar")
ax.set_title("BipedalWalker-v3: does observation-augmentation help under delay?")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_augmentation_ablation.pdf")
fig.savefig(FIG_DIR / "fig4_augmentation_ablation.png")
plt.close(fig)

print("Figures written to", FIG_DIR)
for f in sorted(FIG_DIR.glob("*.png")):
    print(" -", f.name)
