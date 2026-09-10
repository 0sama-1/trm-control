#!/usr/bin/env python3
"""Figures for the mechanistic analysis (analyze_mechanism.py).

  fig8_learning_curves      -- avg100 return vs episode, per trunk, 3 environments
  fig9_critic_loss_curves   -- critic loss vs episode (log y), per trunk, 3 environments
  fig10_capacity_vs_valuefit -- parameter count vs critic-loss ratio to the DNN
                                baseline: does the value-fit deficit track
                                capacity or recursion?
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIG_DIR = PROJECT / "papers" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

summary = json.load(open(HERE / "analysis_mechanism.json"))
curves = json.load(open(HERE / "mechanism_curves.json"))
cells = summary["cells"]

PANELS = [
    ("Pendulum-v1", "fixed-aug", "Pendulum-v1"),
    ("MountainCarContinuous-v0", "fixed-aug", "MountainCarContinuous-v0"),
    ("BipedalWalker-v3", "fixed-aug", "BipedalWalker-v3"),
]
TRUNKS = ["dnn", "lstm", "transformer", "trm", "trm_hier6", "trm_singlez"]
LABELS = {"dnn": "DNN (135k)", "lstm": "LSTM (167k)", "transformer": "Transformer (797k)",
          "trm": "TRM (788k)", "trm_hier6": "TRM-hier6 (794k)", "trm_singlez": "TRM-singlez (794k)"}
COLORS = {"dnn": "#4C72B0", "lstm": "#C44E52", "transformer": "#DD8452",
          "trm": "#55A868", "trm_hier6": "#8172B2", "trm_singlez": "#937860"}
# recursive trunks drawn solid, non-recursive dashed, so the visual grouping
# tests "recursion" vs "capacity" at a glance
STYLE = {"dnn": "--", "lstm": "--", "transformer": "--",
         "trm": "-", "trm_hier6": "-", "trm_singlez": "-"}

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def smooth(y, w=9):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def panel_plot(field, ylabel, title, fname, logy=False, do_smooth=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (env, cond, envlabel) in zip(axes, PANELS):
        for t in TRUNKS:
            key = f"{env}__{cond}__{t}"
            if key not in curves:
                continue
            y = curves[key].get(field) or []
            y = [v for v in y]
            if not y:
                continue
            yy = smooth(y) if do_smooth else np.asarray(y, dtype=float)
            ax.plot(np.arange(1, len(yy) + 1), yy, STYLE[t], color=COLORS[t],
                    label=LABELS[t], linewidth=1.4, alpha=0.9)
        ax.set_title(envlabel, fontsize=9.5)
        ax.set_xlabel("training episode")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.15, linewidth=0.5)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(FIG_DIR / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{fname}.png", bbox_inches="tight")
    plt.close(fig)


panel_plot("avg100_return", "return (100-ep moving average)",
           "Learning curves under 1-step delay with observation augmentation (mean over seeds).  Solid = recursive trunks, dashed = non-recursive.",
           "fig8_learning_curves", do_smooth=False)

panel_plot("critic_loss", "critic loss (log scale)",
           "Critic (value) loss during training (mean over seeds, smoothed).  Recursive trunks sit consistently above the DNN baseline; the Transformer diverges on MountainCarContinuous-v0 and the LSTM on Pendulum-v1.",
           "fig9_critic_loss_curves", logy=True, do_smooth=True)

# --- Figure 10: value-fit deficit by trunk, ordered by capacity -------------
# Categorical strip plot: one point per (environment, condition) cell, a median
# bar per trunk, parity line at 1.0. Two panels (early / late training) because
# late-training critic loss is contaminated by outright divergence on a few
# (trunk, environment) pairs, which the early window is not.
ORDER = ["dnn", "lstm", "trm", "trm_hier6", "trm_singlez", "transformer"]
PARAMS = {"dnn": "135k", "lstm": "167k", "trm": "788k",
          "trm_hier6": "794k", "trm_singlez": "794k", "transformer": "797k"}
RECURSIVE = {"trm", "trm_hier6", "trm_singlez"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (field, phase) in zip(axes, [("critic_loss_early", "early training (first 25% of budget)"),
                                     ("critic_loss_late", "late training (final 25% of budget)")]):
    for i, t in enumerate(ORDER):
        ys = []
        for key, c in cells.items():
            if c["model"] != t:
                continue
            dnn_key = f"{c['env']}__{c['condition']}__dnn"
            if dnn_key not in cells:
                continue
            try:
                a, b = float(c[field]), float(cells[dnn_key][field])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(a) or not np.isfinite(b) or b == 0:
                continue
            ys.append(a / b)
        if not ys:
            continue
        jitter = (np.random.RandomState(0).rand(len(ys)) - 0.5) * 0.26
        ax.scatter(np.full(len(ys), i) + jitter, ys, s=42,
                   color=COLORS[t], marker="o" if t in RECURSIVE else "s",
                   edgecolor="white", linewidth=0.6, zorder=3, alpha=0.95)
        med = float(np.median(ys))
        ax.plot([i - 0.3, i + 0.3], [med, med], color=COLORS[t], linewidth=2.4, zorder=4)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f"{LABELS[t].split(' (')[0]}\n{PARAMS[t]}" for t in ORDER], fontsize=8)
    ax.set_title(phase, fontsize=9.5)
    ax.grid(alpha=0.15, linewidth=0.5, axis="y")
axes[0].set_ylabel("critic loss relative to DNN baseline\n(log scale; 1.0 = parity)")
axes[0].annotate("worse value fit\nthan DNN", xy=(0.02, 0.93), xycoords="axes fraction",
                 fontsize=7.5, color="#555", va="top")
fig.suptitle("Value-fit deficit by trunk, trunks ordered by parameter count.  "
             "Circles = recursive, squares = non-recursive; bars = median across environments/conditions.",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG_DIR / "fig10_capacity_vs_valuefit.pdf")
fig.savefig(FIG_DIR / "fig10_capacity_vs_valuefit.png")
plt.close(fig)

print("Figures written to", FIG_DIR)
for f in sorted(FIG_DIR.glob("fig[89]*.png")) + sorted(FIG_DIR.glob("fig10*.png")):
    print(" -", f.name)
