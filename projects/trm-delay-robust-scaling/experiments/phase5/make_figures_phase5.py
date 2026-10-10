"""Figures for the Phase 5 write-ups -> phase5/report/figures/.

  python3 make_figures_phase5.py

Reads only analysis JSON files (experiments/analysis_p5_t2.json; analysis_s1.json and
analysis_p4_t.json for the Phase 3 / Stage T reference contrasts; phase4 derived_numbers.json
for the Phase 4 run count in the project map), the gates.json files (gate outcomes, the
uniform-random baseline) and the raw per-run results (phase5/results/results_p5t2/ and the
reused Stage T runs in phase4/results/results_p4t/). Writes each figure as PNG + PDF, and
phase5/report/figures/derived_numbers_p5.json: every number the Phase 5 documents quote that
is computed here from raw results rather than read from an analysis JSON. Everything in that
file is exploratory/descriptive; the gate outcome is the one recorded in gates.json.

Colour: family, not arm -- MLP blue, TRM orange (the validated pair used for Phase 4).
Actor learning rate is carried by line style / marker and by axis labels.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
EXP = PROJ / "experiments"
RES5 = PROJ / "phase5" / "results" / "results_p5t2"
RES4 = PROJ / "phase4" / "results" / "results_p4t"
OUT = PROJ / "phase5" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

A = json.loads((EXP / "analysis_p5_t2.json").read_text())
S1 = json.loads((EXP / "analysis_s1.json").read_text())
T4 = json.loads((EXP / "analysis_p4_t.json").read_text())
G5 = json.loads((HERE / "gates.json").read_text())
G4 = json.loads((EXP / "phase4" / "gates.json").read_text())
D4 = json.loads((PROJ / "phase4" / "report" / "figures" / "derived_numbers.json").read_text())
GATE = {g["id"]: g["status"] for g in G5["gates"]}
THR = G5["frozen_values"]["learned_threshold"]
RANDOM = G4["frozen_values"]["span"]["untrained"]["Pendulum-v1"]

MLP, TRM = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.35, "grid.color": GRID, "figure.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2.0})

CONDS = {"C0": "none", "C1": "fixed_aug1"}
# arm key (as in analysis_p5_t2.json) -> (model, lr, source folder, filename tag, label, colour, line style)
ARMS = {
    "trm|1e-4": ("trm", "1e-4", RES5, "_alr1e-4", "TRM 788k, actor LR 1e-4", TRM, "-"),
    "trm|3e-4": ("trm", "3e-4", RES5, "_alr3e-4", "TRM 788k, actor LR 3e-4", TRM, "--"),
    "trm|1e-3": ("trm", "1e-3", RES4, "", "TRM 788k, actor LR 1e-3 (Stage T)", TRM, ":"),
    "dnn_large|1e-4": ("dnn_large", "1e-4", RES5, "_alr1e-4", "MLP 788k, actor LR 1e-4", MLP, "-"),
    "dnn_large|1e-3 (Stage T)": ("dnn_large", "1e-3", RES4, "", "MLP 788k, actor LR 1e-3 (Stage T)", MLP, ":"),
}
DERIVED: dict = {"_note": "Computed by experiments/phase5/make_figures_phase5.py from the raw results files. "
                          "Exploratory/descriptive; the gate outcome is the one recorded in experiments/phase5/gates.json."}


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def sign_p(diffs: list[float]) -> dict:
    nz = [d for d in diffs if d != 0]
    n, k = len(nz), sum(d > 0 for d in nz)
    tail = lambda a: sum(math.comb(n, i) for i in range(a, n + 1)) / 2 ** n
    return {"n": n, "positive": k, "p_two_sided": round(min(1.0, 2 * min(tail(k), tail(n - k))), 4)}


def load_runs() -> dict:
    runs = {}
    for key, (m, lr, folder, tag, *_rest) in ARMS.items():
        for c, ctag in CONDS.items():
            for s in range(1, 7):
                d = json.loads((folder / f"Pendulum-v1_td3_{m}_{ctag}{tag}_seed{s}.json").read_text())
                h = d["episode_history"]
                cl = [r["critic_loss"] for r in h if "critic_loss" in r]
                ev = [r["eval_return"] for r in h if "eval_return" in r]
                runs[(key, c, s)] = {
                    "final": d["final_avg100_return"], "params": d["param_count"], "device": d["device_name"],
                    "wall_min": d["wall_clock_seconds"] / 60, "actor_lr": d["config"]["arch"]["actor_lr"],
                    "critic_lr": d["config"]["arch"]["critic_lr"],
                    "avg100": [r["avg100_return"] for r in h], "critic": cl,
                    "critic_early": st.median(cl[50:75]), "critic_late": st.median(cl[int(0.75 * len(cl)):]),
                    "eval_last5": st.mean(ev[-5:])}
    return runs


def derive(runs: dict) -> None:
    out = {"per_arm": {}, "pairs": {}}
    for key in ARMS:
        rs = {(c, s): v for (k, c, s), v in runs.items() if k == key}
        # self-check against the analysis JSON
        a = A["arms"][key]
        assert abs(st.median(v["critic_late"] for v in rs.values()) - a["critic_late_median"]) < 0.01, key
        assert sum(v["final"] > THR for v in rs.values()) == a["learned"], key
        out["per_arm"][key] = {
            "actor_lr": sorted({v["actor_lr"] for v in rs.values()}), "critic_lr": sorted({v["critic_lr"] for v in rs.values()}),
            "params": {c: sorted({v["params"] for (cc, s), v in rs.items() if cc == c}) for c in CONDS},
            "learned_by_condition": {c: sum(v["final"] > THR for (cc, s), v in rs.items() if cc == c) for c in CONDS},
            "median_final_by_condition": {c: round(st.median(v["final"] for (cc, s), v in rs.items() if cc == c), 2) for c in CONDS},
            "critic_loss_median_ep51_75": round(st.median(v["critic_early"] for v in rs.values()), 1),
            "critic_loss_median_last_quarter": round(st.median(v["critic_late"] for v in rs.values()), 1),
            "runs_critic_late_gt_early": sum(v["critic_late"] > v["critic_early"] for v in rs.values()),
            "devices": {dv: sum(v["device"] == dv for v in rs.values()) for dv in sorted({v["device"] for v in rs.values()})},
            "wall_min_median_L40S": (round(st.median(v["wall_min"] for v in rs.values() if v["device"] == "NVIDIA L40S"), 2)
                                     if any(v["device"] == "NVIDIA L40S" for v in rs.values()) else None),
            "n_L40S": sum(v["device"] == "NVIDIA L40S" for v in rs.values()),
        }

    def pair(a: str, b: str, field: str, ratio: bool = False) -> dict:
        ks = [(c, s) for c in CONDS for s in range(1, 7)]
        if ratio:
            r = [runs[(a, *k)][field] / runs[(b, *k)][field] for k in ks]
            return {"median_ratio": round(st.median(r), 2), "n_above_1": sum(x > 1 for x in r), "n": len(r)}
        d = [runs[(a, *k)][field] - runs[(b, *k)][field] for k in ks]
        return {"median_diff": round(st.median(d), 2), "diffs": [round(x, 2) for x in d], **sign_p(d)}

    P = out["pairs"]
    P["critic_late_ratio trm@1e-4 / dnn_large@1e-4"] = pair("trm|1e-4", "dnn_large|1e-4", "critic_late", True)
    P["critic_late_ratio trm@3e-4 / dnn_large@1e-4"] = pair("trm|3e-4", "dnn_large|1e-4", "critic_late", True)
    P["eval_last5 trm@1e-4 - dnn_large@1e-4"] = pair("trm|1e-4", "dnn_large|1e-4", "eval_last5")
    P["eval_last5 trm@3e-4 - dnn_large@1e-4"] = pair("trm|3e-4", "dnn_large|1e-4", "eval_last5")
    P["final dnn_large@1e-4 - dnn_large@1e-3 (Stage T)"] = pair("dnn_large|1e-4", "dnn_large|1e-3 (Stage T)", "final")
    P["final trm@1e-4 - trm@3e-4"] = pair("trm|1e-4", "trm|3e-4", "final")
    l40 = out["per_arm"]
    out["wall_ratio_L40S trm@1e-4 / dnn_large@1e-4"] = round(l40["trm|1e-4"]["wall_min_median_L40S"]
                                                            / l40["dnn_large|1e-4"]["wall_min_median_L40S"], 1)
    out["phase3_G1_reference"] = {k: S1["contrasts"]["large: trm - dnn_large"][k] for k in ["median_diff", "mean_diff", "pooled_sd"]}
    out["phase3_G1_reference"]["positive"] = S1["contrasts"]["large: trm - dnn_large"]["sign"]["positive"]
    out["stage_t_critic_ratio_trm_dnn_large"] = T4["contrasts"]["critic-late ratio trm/dnn_large"]
    out["result_files_new"] = len(list(RES5.glob("*.json")))
    DERIVED["t2"] = out


def f1_t2(runs: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), gridspec_kw={"width_ratios": [0.9, 1.2, 1.0]})
    fig.subplots_adjust(wspace=0.36)
    order = list(ARMS)
    ticks = ["TRM\n1e-4", "TRM\n3e-4", "TRM\n1e-3*", "MLP\n1e-4", "MLP\n1e-3*"]
    # (a) learned counts
    ax = axes[0]
    for i, k in enumerate(order):
        n = A["arms"][k]["learned"]
        ax.bar(i, n, width=0.62, color=ARMS[k][5], edgecolor="white", lw=2, alpha=0.95 if ARMS[k][1] != "1e-3" else 0.55)
        ax.text(i, n + 0.25, f"{n}/12", ha="center", fontsize=8.5, color=INK)
    ax.axhline(10, color=MUTED, lw=1, ls="--")
    ax.text(4.45, 10.15, "GT2 bar\n(>= 10/12)", fontsize=7.3, color=MUTED, ha="right", va="bottom")
    ax.set_ylim(0, 13.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylabel("runs learned (final avg-100 > -1000)")
    ax.set_title("(a) Runs learned, by actor LR", fontsize=9.5, loc="left")
    ax.grid(axis="x", visible=False)
    # (b) finals per run
    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        for c, mk in [("C0", "o"), ("C1", "s")]:
            ys = [runs[(k, c, s)]["final"] for s in range(1, 7)]
            ax.scatter(i + rng.uniform(-0.12, 0.12, 6), ys, s=30, marker=mk, color=ARMS[k][5], edgecolor="white", lw=0.6, zorder=3)
        m = A["arms"][k]["median_final"]
        ax.plot([i - 0.3, i + 0.3], [m, m], color=INK, lw=1.6, zorder=4)
    ax.axhline(THR, color=MUTED, lw=1, ls="--")
    ax.axhline(RANDOM, color=MUTED, lw=1, ls=":")
    ax.text(4.55, THR + 15, "learned cut -1000", fontsize=7.3, color=MUTED, ha="right", va="bottom")
    ax.text(4.55, RANDOM - 15, "uniform-random -1164", fontsize=7.3, color=MUTED, ha="right", va="top")
    ax.scatter([], [], marker="o", color=MUTED, label="C0 (no delay)")
    ax.scatter([], [], marker="s", color=MUTED, label="C1 (d = 1 + queue)")
    ax.plot([], [], color=INK, lw=1.6, label="median")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=7.5, frameon=False)
    ax.set_xticks(range(5))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_xlim(-0.5, 4.6)
    ax.set_ylabel("final avg-100 training return (TD3)")
    ax.set_title("(b) Final return per run (12 per arm)", fontsize=9.5, loc="left")
    # (c) paired differences TRM 788k - MLP 788k
    ax = axes[2]
    sets = [("DDPG\n(Phase 3 G1)", S1["contrasts"]["large: trm - dnn_large"]["diffs"], "#8a8985"),
            ("TD3, both\nLR 1e-4", A["contrasts"]["trm@1e-4 - dnn_large@1e-4"]["diffs"], TRM),
            ("TD3, TRM 3e-4\nMLP 1e-4", A["contrasts"]["trm@3e-4 - dnn_large@1e-4"]["diffs"], TRM)]
    lo = -260
    for i, (lab, ds, col) in enumerate(sets):
        shown = [max(d, lo + 8) for d in ds]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(ds)), shown, s=30, color=col, edgecolor="white", lw=0.6, zorder=3,
                   alpha=0.55 if i == 2 else 1.0)
        for d in ds:
            if d < lo:
                ax.annotate(f"{d:.0f}", xy=(i, lo + 8), xytext=(i + 0.18, lo + 30), fontsize=7.3, color=MUTED,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
        md = st.median(ds)
        ax.plot([i - 0.3, i + 0.3], [md, md], color=INK, lw=1.6, zorder=4)
        ax.text(i + 0.33, md, f"{md:.1f}", fontsize=7.6, color=INK, va="center")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(lo, 40)
    ax.set_xlim(-0.5, 2.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s[0] for s in sets], fontsize=8)
    ax.set_ylabel("TRM 788k - MLP 788k, per (seed, condition)")
    ax.set_title("(c) Paired difference, n = 12", fontsize=9.5, loc="left")
    fig.suptitle(f"Stage T2 (Pendulum, TD3, only the actor LR changed): GT2 \"{GATE['GT2']}\". "
                 "* = Stage T's runs (TRM reused under P5-1)", fontsize=10, x=0.01, ha="left", y=1.03)
    save(fig, "p5_f1_t2_learned_finals")


def f2_curves(runs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    fig.subplots_adjust(wspace=0.28)
    for ax, field, ylab, title in [(axes[0], "avg100", "avg-100 training return", "(a) Median learning curve (12 runs per arm)"),
                                   (axes[1], "critic", "critic loss (log scale)", "(b) Median critic loss per episode")]:
        for k, (_m, _lr, _f, _t, lab, col, ls) in ARMS.items():
            rs = [runs[(k, c, s)][field] for c in CONDS for s in range(1, 7)]
            L = min(len(r) for r in rs)
            med = [st.median(r[i] for r in rs) for i in range(L)]
            x = np.arange(1, L + 1)
            if field == "critic":
                x, med = x[1:], med[1:]
            ax.plot(x, med, color=col, ls=ls, lw=1.7, label=lab)
        ax.axvline(50, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("episode")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5, loc="left")
    axes[0].text(53, -1450, "end of 10k-step\nrandom warmup", fontsize=7.5, color=MUTED)
    axes[1].set_yscale("log")
    axes[0].legend(loc="upper left", fontsize=7.4, frameon=False)
    fig.suptitle("Stage T2: TRM 788k learns at actor LR 1e-4 and 3e-4; at 1e-3 its curve falls and its critic loss rises. "
                 "The MLP is the same at 1e-4 and 1e-3.", fontsize=9.8, x=0.01, ha="left", y=1.03)
    save(fig, "p5_f2_t2_curves")


def f3_project_map() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_axis_off()
    p4 = D4["result_files"]["training_runs_total"]
    n5 = DERIVED["t2"]["result_files_new"]
    phases = [
        ("Phase 0", "pilot, augmentation fix,\nTRM port, scoping,\nliterature, H1", "groundwork"),
        ("Phase 1", "Pendulum,\nMountainCar;\n4 trunks, 48 runs", "TRM < MLP 0/6;\nH1 threshold\nfalsified"),
        ("Phase 2", "BipedalWalker\n6 trunks, 54 runs", "no difference;\n2 of 3 rungs floor"),
        ("Phase 3", "confound removal\n198 runs, 300\nckpt evaluations", "deficit architectural;\nH3 rejected"),
        ("Phase 4", f"{p4} runs; rung, LR,\ntraining discipline,\nTD3, Lipschitz,\nexport", "GA fail, GC insens.,\nGT pass*,\nGB0/KB0 fail"),
        ("Phase 5", f"{n5} runs; TD3\nactor-LR sweep\nfor TRM 788k", "GT2: LR explains;\nat 1e-4: TRM -\nMLP -13.0 (2/12)"),
    ]
    W, GAP = 0.145, 0.025
    for i, (t, sub, res) in enumerate(phases):
        x = 0.005 + i * (W + GAP)
        ax.add_patch(plt.Rectangle((x, 0.30), W, 0.62, facecolor="#f6f5f2", edgecolor=MUTED, lw=0.8, transform=ax.transAxes))
        ax.text(x + 0.008, 0.87, t, transform=ax.transAxes, weight="bold", va="top", fontsize=9.5, color=INK)
        ax.text(x + 0.008, 0.74, sub, transform=ax.transAxes, va="top", fontsize=7.3, color=INK)
        ax.text(x + 0.008, 0.42, res, transform=ax.transAxes, va="center", fontsize=7.1, color=INK, weight="bold")
        if i < len(phases) - 1:
            ax.annotate("", xy=(x + W + GAP, 0.61), xytext=(x + W, 0.61), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(0.005, 0.12, "* GT passed on sign by optimisation failure (TRM 788k learned 2/12 under td3.yaml); Phase 5 traced that "
            "failure to the actor learning rate.", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.text(0.005, 0.02, "H1 (complexity scaling) is untested across the project: no rung above Pendulum was learned by the MLP baseline.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "p5_f3_project_map")


if __name__ == "__main__":
    R = load_runs()
    derive(R)
    (OUT / "derived_numbers_p5.json").write_text(json.dumps(DERIVED, indent=1, default=str) + "\n")
    print("wrote derived_numbers_p5.json")
    f1_t2(R)
    f2_curves(R)
    f3_project_map()
