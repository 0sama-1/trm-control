#!/usr/bin/env python3
"""Figures for papers/summary.{md,tex,pdf}.

  fig11_command_rate_vs_return -- median |delta a| per step vs median final return,
                                  per (trunk, condition) cell, from
                                  analysis_control_quality.json (no new compute)
  fig12_phase3_stage_map       -- the gated Phase 3 stage plan (papers/phase3/plan.md §4)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "papers" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Same trunk colours/labels as make_figures_mechanism.py so figures 8-12 read as one set.
LABELS = {"dnn": "DNN (135k)", "lstm": "LSTM (167k)", "transformer": "Transformer (797k)",
          "trm": "TRM (788k)", "trm_hier6": "TRM-hier6 (794k)", "trm_singlez": "TRM-singlez (794k)"}
COLORS = {"dnn": "#4C72B0", "lstm": "#C44E52", "transformer": "#DD8452",
          "trm": "#55A868", "trm_hier6": "#8172B2", "trm_singlez": "#937860"}
COND_MARKERS = {"none": "o", "fixed-aug": "s", "fixed-noaug": "^"}
COND_LABELS = {"none": "no delay", "fixed-aug": "1-step delay + aug", "fixed-noaug": "1-step delay, no aug"}


def fig11():
    cq = json.load(open(HERE / "analysis_control_quality.json"))
    cells = cq["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, env in zip(axes, ["Pendulum-v1", "BipedalWalker-v3"]):
        for key, c in cells.items():
            e, cond, trunk = key.split("__")
            if e != env:
                continue
            ax.scatter(c["action_rate_median"], c["return_median"], s=70,
                       color=COLORS[trunk], marker=COND_MARKERS[cond],
                       edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("command rate: median |a$_t$ − a$_{t-1}$| (log scale; left = smoother)")
        ax.set_title(env + (" (every trunk but the LSTM learned)" if env == "Pendulum-v1"
                            else " (floor effect: no run learned to walk)"), fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("median final return (avg of last 100 episodes)")
    axes[0].annotate("LSTM: smoothest, but untrained\n(≈ −1460 = random policy)",
                     xy=(0.0169, -1460), xytext=(0.03, -1100), fontsize=8,
                     arrowprops=dict(arrowstyle="->", color="#555"), color="#333")
    trunk_handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[t], label=LABELS[t], markersize=8)
                     for t in LABELS]
    cond_handles = [plt.Line2D([], [], marker=m, linestyle="", color="#666", label=COND_LABELS[c], markersize=8)
                    for c, m in COND_MARKERS.items()]
    fig.legend(handles=trunk_handles + cond_handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("Command regularity vs. return from the banked Phase 1–2 training data "
                 "(exploration-noisy actions; each point = median over 3 seeds)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig11_command_rate_vs_return.{ext}", dpi=200)
    plt.close(fig)


def fig12():
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, body, face):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=face, edgecolor="#333", linewidth=1.1))
        ax.text(x + 0.15, y + h - 0.22, title, fontsize=10, fontweight="bold", va="top")
        ax.text(x + 0.15, y + h - 0.62, body, fontsize=8.3, va="top", linespacing=1.35)

    def stop(x, y, text):
        ax.add_patch(FancyBboxPatch((x, y), 3.6, 0.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#F4F4F4", edgecolor="#999", linewidth=0.9, linestyle="--"))
        ax.text(x + 0.12, y + 0.58, text, fontsize=7.9, va="top", color="#333", linespacing=1.3)

    def arrow(x0, y0, x1, y1, label=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#333", linewidth=1.1))
        if label:
            ax.text((x0 + x1) / 2 + 0.08, (y0 + y1) / 2, label, fontsize=8, color="#333", va="center")

    committed, cond = "#DCE8F5", "#EFEFEF"
    W, X = 5.9, 0.2
    box(X, 8.55, W, 1.3, "Stage 0 — headroom probe + checkpoint battery  [committed]",
        "18 MLP jobs (fixed vs stochastic vs delay-oracle)  +  0-training\n"
        "re-evaluation of the 102 banked checkpoints: command quality,\nrobustness sweeps, inference latency.   ≈30 CPU-h", committed)
    box(X, 6.75, W, 1.3, "Stage 1 — capacity × depth factorial",
        "DNN 135k / 788k  ×  TRM 135k / 788k,  plus TRM H_cycles ∈ {1, 6}\n"
        "Pendulum-v1, 2 conditions, 6 seeds = 72 jobs\n≈60 CPU-h + 25 GPU-h", cond)
    box(X, 4.95, W, 1.3, "Stage 2 — uncertainty × memory (decisive test)",
        "DNN, TRM Reset-Z, TRM Carry-Z, LSTM  ×  {fixed+aug, stoch+aug, stoch}\n"
        "6 seeds = 72 jobs (+24 time-varying)\n≈70 CPU-h + 40 GPU-h", cond)
    box(X, 3.15, W, 1.3, "Stage 3 — does it generalise?",
        "TD3 cross-check (24)  +  InvertedDoublePendulum-v5\n"
        "budget pilot (9) then matrix (48)\n≈75 GPU-h", cond)
    box(X, 1.35, W, 1.3, "Stage 4 — restore TRM's training discipline",
        "no gradient truncation  /  per-cycle (deep) supervision  /\n"
        "DEQ fixed-point readout;  36 jobs\n≈45 GPU-h", cond)
    ax.text(X + 0.15, 0.75, "Stage 5 — synthesis & report revision (no compute, always runs)",
            fontsize=9.5, fontweight="bold")

    cx = X + W / 2
    arrow(cx, 8.55, cx, 8.05, "always (not gated by G0)")
    arrow(cx, 6.75, cx, 6.25, "G1 sets capacity level;  requires G0 pass")
    arrow(cx, 4.95, cx, 4.45, "G2: interaction flips sign")
    arrow(cx, 3.15, cx, 2.65, "G3: Carry-Z > Reset-Z")
    arrow(cx, 1.35, cx, 1.0)

    sx = 6.7
    stop(sx, 8.85, "G0 fails: no headroom — Stage 2 void.\nPhase ends after Stage 1; write up.")
    stop(sx, 7.05, "G1 has no kill branch: either result\n(capacity or architecture) restates §6.1.")
    stop(sx, 5.25, "G2 fails: H3 rejected. Report the\nnegative with all 3 confounds removed.")
    stop(sx, 3.45, "G3 fails: Stage 4 skipped; P1 loses\nits constructive test.")
    for y in (9.2, 7.4, 5.6, 3.8):
        ax.annotate("", xy=(sx, y), xytext=(X + W, y),
                    arrowprops=dict(arrowstyle="-|>", color="#999", linewidth=0.9, linestyle="--"))
    ax.text(0.2, 0.2, "Total if every gate passes: 279 training jobs (+24 optional) + ~102 evaluation jobs, "
            "≈290 CPU-h + 185 GPU-h.  Only Stage 0 is committed at sign-off.",
            fontsize=8.3, color="#333")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig12_phase3_stage_map.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig11()
    fig12()
    print("wrote fig11_command_rate_vs_return, fig12_phase3_stage_map")
