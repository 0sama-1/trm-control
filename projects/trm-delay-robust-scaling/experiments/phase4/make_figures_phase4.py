"""Figures for the Phase 4 write-ups -> phase4/report/figures/.

  python3 make_figures_phase4.py

Reads only the Phase 4 analysis outputs (experiments/analysis_p4_*.json), gates.json
(for gate outcomes) and the raw per-run results under phase4/results/. Writes each
figure as PNG + PDF, and phase4/report/figures/derived_numbers.json: every number the
Phase 4 documents quote that is computed here from raw results rather than read from
an analysis JSON (solve fractions, nested-budget equality, Stage T learning counts and
critic-loss windows, Stage A clip-active fractions, parameter counts, Stage E pairings).

Colour: family, not trunk -- MLP blue, TRM orange (validated pair). Identity within a
family is carried by axis labels, marker shape or line style.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
EXP = PROJ / "experiments"
RES = PROJ / "phase4" / "results"
OUT = PROJ / "phase4" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

A = {k: json.loads((EXP / f"analysis_p4_{k}.json").read_text()) for k in ["b0", "b0h", "b0d", "c", "a0", "a", "t", "e"]}
G = json.loads((HERE / "gates.json").read_text())
GATE = {g["id"]: g["status"] for g in G["gates"]}

MLP, TRM = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
LBL = {"dnn": "MLP 135k", "dnn_large": "MLP 788k", "trm_small": "TRM 136k", "trm": "TRM 788k (H=3)",
       "trm_h1": "TRM H=1", "trm_h6": "TRM H=6"}
FAM = lambda m: MLP if m.startswith("dnn") else TRM

plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.35, "grid.color": GRID, "figure.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2.0})

DERIVED: dict = {"_note": "Computed by experiments/phase4/make_figures_phase4.py from phase4/results/ raw files. "
                          "Exploratory/descriptive unless stated; gate outcomes are in gates.json."}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def jitter(n: int, w: float = 0.10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-w, w, n)


# ---------------------------------------------------------------------------
# derived numbers from raw results
# ---------------------------------------------------------------------------

def avg100_at(history: list[dict], budget: int):
    """Same rule as gates_eval.avg100_at: avg-100 after the last episode that started before `budget`."""
    prev, val = 0, None
    for row in history:
        if prev >= budget:
            break
        val, prev = row["avg100_return"], row["total_steps"]
    return val


def derive_b0() -> None:
    out = {}
    for tag, folder, env in [("idp", "results_p4b0", "InvertedDoublePendulum-v5"), ("hopper", "results_p4b0h", "Hopper-v5")]:
        runs = {}
        for b in [150, 300, 500]:
            for s in [1, 2, 3]:
                d = load(RES / folder / f"{env}_td3_dnn_fixed_aug1_{b}k_seed{s}.json")
                h = d["episode_history"]
                lens = [r["total_steps"] - (h[i - 1]["total_steps"] if i else 0) for i, r in enumerate(h)]
                last = h[-100:]
                runs[f"{b}k_s{s}"] = {"device": d["device_name"], "final": d["final_avg100_return"], "history": h,
                                      "params": d["param_count"],
                                      "last100_return_gt_9000": sum(r["return"] > 9000 for r in last),
                                      "last100_len_1000": sum(l == 1000 for l in lens[-100:])}
        nested = []
        for s in [1, 2, 3]:
            big = runs[f"500k_s{s}"]
            for b in [150, 300]:
                small = runs[f"{b}k_s{s}"]
                v = avg100_at(big["history"], b * 1000)
                nested.append({"seed": s, "budget_k": b, "avg100_in_500k_run": round(v, 4),
                               "final_of_budget_run": round(small["final"], 4), "identical": v == small["final"],
                               "device_budget_run": small["device"], "device_500k_run": big["device"]})
        same = [x for x in nested if x["device_budget_run"] == x["device_500k_run"]]
        diff = [x for x in nested if x["device_budget_run"] != x["device_500k_run"]]
        out[tag] = {"nested_check": nested,
                    "nested_same_gpu_model_identical": f"{sum(x['identical'] for x in same)}/{len(same)}",
                    "nested_different_gpu_model_identical": f"{sum(x['identical'] for x in diff)}/{len(diff)}",
                    "params_actor_plus_twin_critics": sorted({r["params"] for r in runs.values()})}
        if tag == "idp":
            out[tag]["last100_full_solves_500k"] = {f"seed{s}": {"return_gt_9000": runs[f"500k_s{s}"]["last100_return_gt_9000"],
                                                                 "length_1000": runs[f"500k_s{s}"]["last100_len_1000"]}
                                                    for s in [1, 2, 3]}
            out[tag]["last100_full_solves_by_budget"] = {k: v["last100_return_gt_9000"] for k, v in runs.items()}
            pooled = [r["return"] for s in [1, 2, 3] for r in runs[f"500k_s{s}"]["history"][-100:]]
            out[tag]["last100_500k_pooled_returns"] = {"n": len(pooled), "gt_9000": sum(x > 9000 for x in pooled),
                                                       "lt_500": sum(x < 500 for x in pooled),
                                                       "between": sum(500 <= x <= 9000 for x in pooled)}
    timing = {}
    for m in ["trm", "trm_h1"]:
        d = load(RES / "results_p4b0" / f"InvertedDoublePendulum-v5_td3_{m}_fixed_aug1_timing15k_seed1.json")
        timing[m] = {"params_actor_plus_twin_critics": d["param_count"], "device": d["device_name"]}
    out["idp"]["timing_runs"] = timing
    d0 = load(RES / "results_p4b0d" / "InvertedDoublePendulum-v5_td3_dnn_none_500k_seed1.json")
    out["idp"]["params_no_delay"] = d0["param_count"]
    DERIVED["b0"] = out


def derive_t() -> dict:
    runs = {}
    for m in ["dnn", "dnn_large", "trm_small", "trm"]:
        for c, tag in [("none", "none"), ("C1", "fixed_aug1")]:
            for s in range(1, 7):
                d = load(RES / "results_p4t" / f"Pendulum-v1_td3_{m}_{tag}_seed{s}.json")
                h = d["episode_history"]
                runs[(m, c, s)] = {"final": d["final_avg100_return"], "params": d["param_count"], "device": d["device_name"],
                                   "avg100": [r["avg100_return"] for r in h],
                                   "critic": [r.get("critic_loss") for r in h]}
    CUT = -1000.0
    out = {"learned_cut": CUT, "learned_cut_note": "post hoc descriptive cut separating learned runs from random-level "
           "runs (uniform-random policy -1163.84, gates.json span.untrained); reproduces the counts in gates.json GT detail",
           "per_trunk": {}}
    for m in ["dnn", "dnn_large", "trm_small", "trm"]:
        rs = {k: v for k, v in runs.items() if k[0] == m}
        early = {f"{k[1]}_s{k[2]}": st.median([x for x in v["critic"][50:75] if x is not None]) for k, v in rs.items()}
        late = {}
        for k, v in rs.items():
            cl = [x for x in v["critic"] if x is not None]
            late[f"{k[1]}_s{k[2]}"] = st.median(cl[int(0.75 * len(cl)):])
        out["per_trunk"][m] = {
            "finals": {f"{k[1]}_s{k[2]}": round(v["final"], 1) for k, v in rs.items()},
            "n_learned": sum(v["final"] > CUT for v in rs.values()), "n": len(rs),
            "params_actor_plus_twin_critics": {c: sorted({v["params"] for k, v in rs.items() if k[1] == c}) for c in ["none", "C1"]},
            "critic_loss_median_ep51_75": round(st.median(early.values()), 1),
            "critic_loss_median_last_quarter": round(st.median(late.values()), 1),
            "runs_critic_late_gt_early": sum(late[k] > early[k] for k in early),
        }
    DERIVED["t"] = out
    return runs


def derive_a() -> dict:
    clip = G["frozen_values"]["stage_a_clip"]
    runs, out = {}, {"clip_norms": {"actor": clip["actor"], "critic": clip["critic"]}, "clip_active_fraction": {}, "params": {}}
    for m in ["trm", "trm_fullgrad", "trm_fullgrad_ds"]:
        for c, tag in [("C1", "fixed_aug1"), ("C4", "stochastic_aug0")]:
            fr_a, fr_c, prm = [], [], set()
            for s in range(1, 7):
                d = load(RES / "results_p4a" / f"Pendulum-v1_ddpg_{m}_{tag}_pa_clip_seed{s}.json")
                h = d["episode_history"]
                an = [x for r in h for x in r.get("actor_grad_norms", [])]
                cn = [x for r in h for x in r.get("critic_grad_norms", [])]
                fr_a.append(sum(x > clip["actor"] for x in an) / len(an))
                fr_c.append(sum(x > clip["critic"] for x in cn) / len(cn))
                prm.add(d["param_count"])
                runs[(m, c, s)] = d["final_avg100_return"]
            out["clip_active_fraction"][f"{m}|{c}"] = {"actor_median": round(st.median(fr_a), 4),
                                                       "actor_range": [round(min(fr_a), 4), round(max(fr_a), 4)],
                                                       "critic_median": round(st.median(fr_c), 4),
                                                       "critic_range": [round(min(fr_c), 4), round(max(fr_c), 4)]}
            out["params"][f"{m}|{c}"] = sorted(prm)
    out["clip_active_note"] = ("fraction of per-update pre-clip gradient norms above the frozen clip norm, per run, "
                               "over all updates; median and range over seeds 1-6")
    c_params = sorted({load(p)["param_count"] for p in (RES / "results_p4c").glob("*.json")})
    out["stage_c_params"] = c_params
    DERIVED["a"] = out
    return runs


def perm_p(x: list[float], y: list[float], n: int = 20000, seed: int = 0) -> float:
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(1, len(v) + 1)
        return r
    rx, ry = rank(np.array(x)), rank(np.array(y))
    obs = abs(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    hits = sum(abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def derive_e() -> None:
    rows = A["e"]["rows"]
    key = lambda r: (r["delay"], r["seed"])
    by = {m: {key(r): r for r in rows if r["model"] == m} for m in {r["model"] for r in rows}}
    out = {"label": "exploratory", "paired": {}, "spearman_permutation_p": {}}
    for a, b in [("trm", "dnn"), ("trm", "dnn_large"), ("trm_small", "dnn")]:
        ks = sorted(set(by[a]) & set(by[b]))
        for metric in ["lip_median", "noise_drop"]:
            d = [by[a][k][metric] - by[b][k][metric] for k in ks]
            out["paired"][f"{a} - {b} | {metric}"] = {"n": len(d), "n_positive": sum(x > 0 for x in d),
                                                     "median_diff": round(st.median(d), 2)}
    fam = lambda m: "MLP" if m.startswith("dnn") else "TRM"
    for scope, rs in [("all", rows), ("MLP", [r for r in rows if fam(r["model"]) == "MLP"]),
                      ("TRM", [r for r in rows if fam(r["model"]) == "TRM"])]:
        out["spearman_permutation_p"][scope] = {
            "n": len(rs),
            "lip_vs_reversals": round(perm_p([r["lip_median"] for r in rs], [r["reversals_per_s"] for r in rs]), 4),
            "lip_vs_noise_drop": round(perm_p([r["lip_median"] for r in rs], [r["noise_drop"] for r in rs]), 4)}
    out["spearman_permutation_note"] = "two-sided permutation test on Spearman rho, 20,000 permutations, numpy seed 0"
    low = [r for r in rows if r["lip_median"] < 5]
    hi = [r for r in low if r["reversals_per_s"] > 5]
    out["low_lipschitz"] = {
        "n_lip_median_lt_5": len(low), "n_of_those_reversals_gt_5": len(hi),
        "those": [{"model": r["model"], "seed": r["seed"], "delay": r["delay"], "lip_median": round(r["lip_median"], 3),
                   "lip_p95": round(r["lip_p95"], 1), "reversals_per_s": round(r["reversals_per_s"], 2),
                   "abs_du": round(r["abs_du"], 3)} for r in sorted(hi, key=lambda r: r["lip_median"])],
        "n_all_reversals_gt_5": sum(r["reversals_per_s"] > 5 for r in rows)}
    DERIVED["e"] = out


def derive_counts() -> None:
    folders = ["results_p4b0", "results_p4b0h", "results_p4b0d", "results_p4c", "results_p4a0", "results_p4a", "results_p4t", "results_p4e"]
    DERIVED["result_files"] = {f: len(list((RES / f).glob("*.json"))) for f in folders}
    DERIVED["result_files"]["training_runs_total"] = sum(v for k, v in DERIVED["result_files"].items() if k != "results_p4e")
    exp = load(RES / "export_c" / "export_summary.json")
    DERIVED["export_c"] = [{"model": r["model"], "actor_params": r["actor_params"], "buffer_floats": r["buffer_floats"],
                            "fp32_KiB": round(r["fp32_bytes"] / 1024, 1), "int8_KiB": round(r["int8_bytes"] / 1024, 1)}
                           for r in exp]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def f1_stage_map() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_axis_off()
    boxes = [
        (0.00, 0.52, "B0: complexity pilot", "IDP-v5, TD3, MLP\n150k/300k/500k (9 + 2 timing)", f"GB0 {GATE['GB0']}"),
        (0.00, 0.05, "B0h / B0d", "Hopper pilot (9); IDP\ndiagnostic 1M / no delay (6)", f"KB0 {GATE['KB0']}"),
        (0.25, 0.52, "B1: complexity rung", "TD3, 4 trunks x 6 seeds\n(24 planned)", "H1' not run (K-3)"),
        (0.25, 0.05, "C: LR sweep", "TRM 788k, 3x3 actor x critic\nLR grid x 3 seeds (27)", f"GC {GATE['GC']}"),
        (0.50, 0.52, "A: training discipline", "pilot (4) + full grad,\n+ deep sup. (36; MLP reused)", f"GA {GATE['GA']}"),
        (0.50, 0.05, "T: G1 under TD3", "4 trunks x 2 conditions\nx 6 seeds (48)", f"GT {GATE['GT']} (caveat)"),
        (0.75, 0.52, "E: Lipschitz", "72 Phase 3 Stage 1\ncheckpoints, no training", "exploratory"),
        (0.75, 0.05, "D: C-header export", "3 actors, FP32 headers,\n20 reference I/O pairs", "engineering"),
    ]
    col = {"fail": "#b23a24", "pass": "#2f7a4a", "insensitive": "#2f7a4a"}
    for x, y, t, sub, gate in boxes:
        notrun = "not run" in gate
        fc = "#eeeeee" if notrun else "#f6f5f2"
        ax.add_patch(plt.Rectangle((x + 0.005, y), 0.225, 0.40, facecolor=fc, edgecolor=MUTED, lw=0.8,
                                   transform=ax.transAxes, ls="--" if notrun else "-"))
        ax.text(x + 0.015, y + 0.35, t, transform=ax.transAxes, weight="bold", va="top", fontsize=9, color=INK)
        ax.text(x + 0.015, y + 0.24, sub, transform=ax.transAxes, va="top", fontsize=7.8, color=INK)
        status = gate.split()[1] if len(gate.split()) > 1 else ""
        gc = col.get(status, MUTED)
        if "caveat" in gate:
            gc = "#8a6d00"
        ax.text(x + 0.015, y + 0.035, gate, transform=ax.transAxes, va="bottom", fontsize=8.2, weight="bold", color=gc)
    ax.annotate("", xy=(0.255, 0.72), xytext=(0.23, 0.72), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTED, ls="--"))
    ax.text(0.0, -0.06, "Verification V-1 (bit-identity 30/30, gradient-hook 7/21 vs 21/21) passed before any B0, C or A job. "
            "Every TRM run is DDPG unless marked TD3.", transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "p4_f1_stage_map")


def f2_b0() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    fig.subplots_adjust(wspace=0.32)
    marks = {"1": "o", "2": "s", "3": "^"}
    for ax, key, title in [(axes[0], "b0", "InvertedDoublePendulum-v5"), (axes[1], "b0h", "Hopper-v5")]:
        a = A[key]
        xs = [150, 300, 500]
        for s in ["1", "2", "3"]:
            ys = [a["budgets"][str(b * 1000)]["finals"][s] for b in xs]
            ax.plot(xs, ys, marker=marks[s], color=MLP, lw=1.2, ms=7, alpha=0.9, label=f"seed {s}")
        if key == "b0":
            c1 = A["b0d"]["arms"]["C1_1000k"]["seeds"]
            for s in ["1", "2", "3"]:
                ax.plot([1000], [c1[s]["final"]], marker=marks[s], color=MLP, mfc="white", ms=7, lw=0)
            ax.text(1000, 5000, "1M\n(K-2)", ha="center", fontsize=7.8, color=MUTED)
            xs_ticks = [150, 300, 500, 1000]
        else:
            xs_ticks = xs
        thr = a["threshold"]
        ax.axhline(thr, color=INK, lw=1, ls="--")
        ax.text(xs_ticks[0], thr * 1.02, f"learned bar {thr}", fontsize=8, color=INK, va="bottom")
        ax.set_xticks(xs_ticks)
        ax.set_xticklabels([f"{x}k" if x < 1000 else "1M" for x in xs_ticks])
        ax.set_xlabel("env-step budget")
        ax.set_ylabel("final avg-100 training return")
        ax.set_title(f"{title}: 0/3 learned at every budget", fontsize=9.5, loc="left")
        ax.set_ylim(0, thr * 1.12)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=8, frameon=False)
    fig.suptitle("Stage B0: the MLP baseline does not reach the learned bar on either rung (TD3, fixed d = 1 + queue)",
                 fontsize=10, x=0.01, ha="left", y=1.02)
    save(fig, "p4_f2_b0_pilots")


def f3_idp() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    ls = {1: "-", 2: "--", 3: ":"}
    for ax, tag, title in [(axes[0], "fixed_aug1_1000k", "C1 (d = 1 + queue), 1M steps"),
                           (axes[1], "none_500k", "no delay, 500k steps")]:
        for s in [1, 2, 3]:
            h = load(RES / "results_p4b0d" / f"InvertedDoublePendulum-v5_td3_dnn_{tag}_seed{s}.json")["episode_history"]
            h = [r for r in h if r["episode"] >= 100]
            ax.plot([r["total_steps"] / 1000 for r in h], [r["avg100_return"] for r in h], color=MLP, ls=ls[s], lw=1.3,
                    label=f"seed {s}")
        ax.axhline(8190, color=INK, lw=1, ls="--")
        ax.text(5, 8350, "8190", fontsize=8)
        ax.set_ylim(0, 9800)
        ax.set_xlabel("env steps (k)")
        ax.set_title(title, fontsize=9.5, loc="left")
    axes[0].set_ylabel("avg-100 training return")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    ax = axes[2]
    rets = []
    for s in [1, 2, 3]:
        h = load(RES / "results_p4b0" / f"InvertedDoublePendulum-v5_td3_dnn_fixed_aug1_500k_seed{s}.json")["episode_history"]
        rets += [r["return"] for r in h[-100:]]
    ax.hist(rets, bins=np.linspace(0, 9500, 39), color=MLP, edgecolor="white", lw=0.6)
    ax.set_xlabel("episode return")
    ax.set_ylabel("episodes")
    ax.set_title("last 100 episodes of the 500k runs\n(3 seeds pooled, C1)", fontsize=9.5, loc="left")
    fig.suptitle("IDP: under one-step delay the MLP plateaus with bimodal episodes; without delay it reaches the bar but does not hold it",
                 fontsize=10, x=0.01, ha="left", y=1.04)
    save(fig, "p4_f3_idp_diagnostic")


def f4_stage_c() -> None:
    m = ["0.3", "1", "3"]
    grid = np.array([[A["c"]["cells"][f"a{a}c{c}"]["median_final"] for c in m] for a in m])
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.grid(False)
    im = ax.imshow(grid, cmap="Oranges", vmin=-290, vmax=-190)
    for i, a in enumerate(m):
        for j, c in enumerate(m):
            cell = A["c"]["cells"][f"a{a}c{c}"]
            txt = f"{cell['median_final']:.1f}"
            if "vs_default" in cell:
                v = cell["vs_default"]
                txt += f"\nΔ {v['median_diff']:+.1f}  ({v['sign']['positive']}/3)"
            else:
                txt += "\ndefault"
            dark = grid[i, j] > -230
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.2, color="white" if dark else INK,
                    weight="bold" if "vs_default" not in cell else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"×{x}\n({1e-3 * float(x):.0e})" for x in m])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"×{x} ({1e-4 * float(x):.0e})" for x in m])
    ax.set_xlabel("critic learning rate")
    ax.set_ylabel("actor learning rate")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("median final avg-100 (darker = better)")
    ax.set_title(f"Stage C: TRM 788k, Pendulum, DDPG, C1\nGC: {GATE['GC']} (no cell flagged)", fontsize=9.5, loc="left")
    save(fig, "p4_f4_stage_c")


def f5_stage_a(runs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    keys = [("trm_fullgrad", "C1"), ("trm_fullgrad", "C4"), ("trm_fullgrad_ds", "C1"), ("trm_fullgrad_ds", "C4")]
    for i, (d, c) in enumerate(keys):
        ct = A["a"]["contrasts"][f"{d} - trm | {c}"]
        ys = ct["diffs"]
        ax.scatter(i + jitter(len(ys), seed=i), ys, s=34, color=TRM, edgecolor="white", lw=0.6, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [ct["median_diff"]] * 2, color=INK, lw=2, zorder=4)
        ax.text(i, 72, f"{ct['sign']['positive']}/6", ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["full grad\nC1", "full grad\nC4", "full grad\n+ deep sup.\nC1", "full grad\n+ deep sup.\nC4"])
    ax.set_ylabel("Δ final avg-100 vs TRM-trunc (per seed)")
    ax.set_ylim(-100, 85)
    ax.set_title(f"GA: positive in k/6 seeds (pass needs ≥ 5/6) -> {GATE['GA']}", fontsize=9.5, loc="left")
    ax = axes[1]
    arms = [("trm", "TRM-\ntrunc"), ("trm_fullgrad", "full\ngrad"), ("trm_fullgrad_ds", "full grad\n+ DS")]
    for k, c in enumerate(["C1", "C4"]):
        base = k * 4
        for i, (m, lab) in enumerate(arms):
            ys = [runs[(m, c, s)] for s in range(1, 7)]
            ax.scatter(base + i + jitter(6, seed=10 + i + k), ys, s=30, color=TRM, edgecolor="white", lw=0.6, zorder=3)
            ax.plot([base + i - 0.22, base + i + 0.22], [st.median(ys)] * 2, color=INK, lw=2, zorder=4)
        mlp = A["a"]["arms"][f"dnn|{c}"]["median_final"]
        ax.plot([base - 0.4, base + 2.4], [mlp, mlp], color=MLP, ls="--", lw=1.5)
        ax.text(base + 2.45, mlp, f"MLP 135k\nmedian {mlp:.0f}", color=INK, fontsize=7.8, va="center")
        ax.text(base + 1, -60, c + (" (fixed d = 1 + queue)" if c == "C1" else " (stochastic, no queue)"),
                ha="center", fontsize=8.5, color=INK)
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels([a[1] for a in arms] * 2, fontsize=8)
    ax.set_ylim(-470, -40)
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylabel("final avg-100 training return")
    ax.set_title("Per-seed returns; all TRM arms clipped (actor 15, critic 2800), MLP reused unclipped (R-1)",
                 fontsize=9, loc="left")
    save(fig, "p4_f5_stage_a")


def f6_stage_t(runs: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]})
    fig.subplots_adjust(wspace=0.38)
    order = ["dnn", "dnn_large", "trm_small", "trm"]
    ax = axes[0]
    for i, m in enumerate(order):
        for c, mk in [("none", "o"), ("C1", "s")]:
            ys = [runs[(m, c, s)]["final"] for s in range(1, 7)]
            ax.scatter(i + jitter(6, seed=i * 3 + (c == "C1")), ys, s=28, marker=mk, color=FAM(m), edgecolor="white",
                       lw=0.5, zorder=3, label=None)
        ddpg = A["t"]["ddpg_stage1_medians"][m]
        ax.plot([i - 0.3, i + 0.3], [ddpg, ddpg], color=INK, lw=1.6, zorder=4)
    ax.axhline(G["frozen_values"]["span"]["untrained"]["Pendulum-v1"], color=MUTED, lw=1, ls=":")
    ax.text(3.6, -1110, "uniform-\nrandom", fontsize=7.5, color=MUTED, va="bottom")
    ax.scatter([], [], marker="o", color=MUTED, label="no delay")
    ax.scatter([], [], marker="s", color=MUTED, label="C1")
    ax.plot([], [], color=INK, lw=1.6, label="DDPG median (Phase 3)")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["MLP\n135k", "MLP\n788k", "TRM\n136k", "TRM\n788k"])
    ax.set_xlim(-0.5, 4.3)
    ax.set_ylabel("final avg-100 training return (TD3)")
    ax.set_title("Final return per run (12 per trunk)", fontsize=9.5, loc="left")
    styles = {"dnn": "-", "dnn_large": "--", "trm_small": "--", "trm": "-"}
    for ax, key, ylab, title in [(axes[1], "avg100", "avg-100 training return", "Median learning curve (12 runs)"),
                                 (axes[2], "critic", "critic loss (log scale)", "Median critic loss per episode")]:
        for m in order:
            rs = [runs[(m, c, s)][key] for c in ["none", "C1"] for s in range(1, 7)]
            L = min(len(r) for r in rs)
            med = [st.median([r[i] for r in rs if r[i] is not None]) for i in range(L) if all(r[i] is not None for r in rs)]
            x = np.arange(L - len(med) + 1, L + 1)
            ax.plot(x, med, color=FAM(m), ls=styles[m], lw=1.6, label=LBL[m].replace(" (H=3)", ""))
        ax.axvline(50, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("episode")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5, loc="left")
    axes[1].text(53, -350, "end of 10k-step\nrandom warmup", fontsize=7.5, color=MUTED)
    axes[2].set_yscale("log")
    axes[1].legend(loc="center right", fontsize=7.8, frameon=False)
    fig.suptitle(f"Stage T (Pendulum, TD3 as in td3.yaml): GT {GATE['GT']}, but TRM 788k mostly fails to train; its critic loss rises",
                 fontsize=10, x=0.01, ha="left", y=1.03)
    save(fig, "p4_f6_stage_t")


def f7_stage_e() -> None:
    rows = A["e"]["rows"]
    marks = {"dnn": "o", "dnn_large": "s", "trm_small": "^", "trm": "o", "trm_h1": "D", "trm_h6": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, ykey, ylab, sp in [(axes[0], "reversals_per_s", "torque direction reversals / s", "lip_vs_reversals"),
                               (axes[1], "noise_drop", "return drop under sensor noise σ = 0.05", "lip_vs_noise_drop")]:
        for m in ["dnn", "dnn_large", "trm_small", "trm", "trm_h1", "trm_h6"]:
            rs = [r for r in rows if r["model"] == m]
            ax.scatter([r["lip_median"] for r in rs], [r[ykey] for r in rs], marker=marks[m], s=30,
                       facecolor=FAM(m) if m in ("dnn", "trm", "trm_h1") else "white", edgecolor=FAM(m), lw=1.2,
                       label=LBL[m].replace(" (H=3)", ""))
        s = A["e"]["spearman"]
        ax.set_xlabel("actor local Lipschitz constant\n(median spectral norm of ∂a/∂s over visited states)")
        ax.set_ylabel(ylab)
        ax.set_title(f"Spearman ρ: all {s['all'][sp]:+.2f}, MLP {s['MLP'][sp]:+.2f}, TRM {s['TRM'][sp]:+.2f}",
                     fontsize=9.5, loc="left")
    axes[0].set_yscale("log")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.8, frameon=False)
    fig.suptitle("Stage E (exploratory): TRM actors are smoother in the Lipschitz sense, yet lose more return under sensor noise",
                 fontsize=10, x=0.01, ha="left", y=1.03)
    save(fig, "p4_f7_stage_e")


def f8_cost() -> None:
    tc = A["a"]["training_cost_median_min_by_gpu"]
    lat = A["a"]["latency_actor_b1_cpu1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(wspace=0.45)
    ax = axes[0]
    arms = [("dnn", "MLP 135k", MLP, ""), ("trm", "TRM-trunc", TRM, ""), ("trm_fullgrad", "full grad", TRM, "///"),
            ("trm_fullgrad_ds", "full grad + deep sup.", TRM, "...")]
    gpus = ["L40S", "V100", "A100"]
    w = 0.2
    for j, (m, lab, c, hatch) in enumerate(arms):
        for i, g in enumerate(gpus):
            if m not in tc[g]:
                continue
            x = i + (j - 1.5) * w
            ax.bar(x, tc[g][m], width=w * 0.9, color=c if not hatch else "white", edgecolor=c, hatch=hatch, lw=1.2,
                   label=lab if i == (1 if m == "dnn" else 0) else None)
            ax.text(x, tc[g][m] + 0.8, f"{tc[g][m]:.1f}", ha="center", fontsize=7.2, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(gpus)
    ax.set_ylabel("median wall-clock per run (min)")
    ax.set_ylim(0, 84)
    ax.legend(loc="upper left", fontsize=7.8, frameon=False, ncol=2)
    ax.set_title("Training cost within GPU model (Stage A, 200 episodes)", fontsize=9.5, loc="left")
    ax = axes[1]
    order = [("dnn", "MLP 135k"), ("trm", "TRM-trunc"), ("trm_fullgrad", "full grad"), ("trm_fullgrad_ds", "full grad + DS")]
    for i, (m, lab) in enumerate(order):
        p50, p99 = lat["p50_ms"][m], lat["p99_ms"][m]
        ax.plot([p50, p99], [i, i], color=FAM(m), lw=2)
        ax.scatter([p50], [i], color=FAM(m), s=40, zorder=3)
        ax.scatter([p99], [i], color="white", edgecolor=FAM(m), s=40, zorder=3, lw=1.4)
        ax.text(p99 * 1.15, i, f"p50 {p50:.2f} / p99 {p99:.2f} ms", fontsize=7.8, va="center", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 400)
    ax.set_yticks(range(4))
    ax.set_yticklabels([o[1] for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("actor forward, batch 1, 1 CPU thread, ms (log)")
    ax.set_title("Decision latency (local workstation; ratios only)", fontsize=9.5, loc="left")
    save(fig, "p4_f8_cost_latency")


def f9_project_map() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_axis_off()
    phases = [
        ("Phase 0", "pilot, augmentation fix,\nTRM port, scoping,\nliterature, H1", "groundwork", "#f6f5f2"),
        ("Phase 1", "Pendulum, MountainCar\n4 trunks, 48 runs", "TRM < MLP 0/6;\nH1 threshold\nfalsified", "#f6f5f2"),
        ("Phase 2", "BipedalWalker\n6 trunks, 54 runs", "no difference;\n2 of 3 rungs floor", "#f6f5f2"),
        ("Phase 3", "confound removal\n198 runs, 300\nckpt evaluations", "deficit architectural;\nH3 rejected", "#f6f5f2"),
        ("Phase 4", f"{DERIVED['result_files']['training_runs_total']} runs, {DERIVED['result_files']['results_p4e']} ckpt\nLipschitz, export", "GA fail, GC insens.,\nGT pass*,\nGB0/KB0 fail", "#f6f5f2"),
    ]
    W, GAP = 0.17, 0.035
    for i, (t, sub, res, c) in enumerate(phases):
        x = 0.005 + i * (W + GAP)
        ax.add_patch(plt.Rectangle((x, 0.30), W, 0.62, facecolor=c, edgecolor=MUTED, lw=0.8, transform=ax.transAxes))
        ax.text(x + 0.01, 0.87, t, transform=ax.transAxes, weight="bold", va="top", fontsize=9.5, color=INK)
        ax.text(x + 0.01, 0.73, sub, transform=ax.transAxes, va="top", fontsize=7.9, color=INK)
        ax.text(x + 0.01, 0.42, res, transform=ax.transAxes, va="center", fontsize=7.6, color=INK, weight="bold")
        if i < 4:
            ax.annotate("", xy=(x + W + GAP, 0.61), xytext=(x + W, 0.61), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(0.005, 0.12, "* GT passes on its rule, but TRM 788k does not train under td3.yaml (2/12 runs learn): an optimisation failure, "
            "not a replication of Phase 3's deficit.", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.text(0.005, 0.02, "H1 (complexity scaling) is untested across the project: no rung above Pendulum was learned by the MLP baseline.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "p4_f9_project_map")


if __name__ == "__main__":
    derive_b0()
    t_runs = derive_t()
    a_runs = derive_a()
    derive_e()
    derive_counts()
    (OUT / "derived_numbers.json").write_text(json.dumps(DERIVED, indent=1, default=str) + "\n")
    print("wrote derived_numbers.json")
    f1_stage_map()
    f2_b0()
    f3_idp()
    f4_stage_c()
    f5_stage_a(a_runs)
    f6_stage_t(t_runs)
    f7_stage_e()
    f8_cost()
    f9_project_map()
