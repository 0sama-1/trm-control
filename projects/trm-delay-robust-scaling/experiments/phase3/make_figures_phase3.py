"""Figures for the Phase 3 write-ups -> phase3/report/figures/.

  python3 make_figures_phase3.py

Reads phase3/results/runs_cache.json and experiments/analysis_phase3_full.json.
Panels that need data not yet present (battery2, latent-delay) are skipped.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
OUT = PROJ / "phase3" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = json.loads((PROJ / "phase3" / "results" / "runs_cache.json").read_text())
A = json.loads((PROJ / "experiments" / "analysis_phase3_full.json").read_text())

COL = {"dnn": "#2f3e46", "dnn_large": "#7d93a8", "trm_small": "#f4a261", "trm": "#e76f51", "trm_h1": "#e9c46a",
       "trm_h6": "#9c4a2f", "trm_carryz": "#2a9d8f", "lstm_carry": "#7b5ea7", "mlp_oracle": "#6c8f3e"}
LBL = {"dnn": "MLP 135k", "dnn_large": "MLP 788k", "trm_small": "TRM 136k", "trm": "TRM 788k (H=3)",
       "trm_h1": "TRM H=1", "trm_h6": "TRM H=6", "trm_carryz": "TRM Carry-Z", "lstm_carry": "LSTM (carried)"}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def cond_of(r):
    if r["delay"] == "none":
        return "none"
    base = {"fixed": "fixed1", "stochastic": "stoch", "episode": "episode"}[r["delay"]]
    s = f"{base}_aug{int(r['aug'])}"
    return s + ("_" + r["tag"] if r["tag"] else "")


def runs(stage, model=None, cond=None):
    return [r for r in RUNS if r["stage"] == stage and (model is None or r["model"] == model)
            and (cond is None or cond_of(r) == cond)]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def jitter(n, w=0.12, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n)


# F1 -- stage map with outcomes ------------------------------------------------
def f1():
    g = {x: A for x in []}
    lat = A["latent_delay"]
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_axis_off()
    W, GAP = 0.155, 0.045
    boxes = [("Stage 0", "headroom probe (24)\nbattery, 102 ckpts\nlatency bench", "#dcefe2"),
             ("Stage 1", "capacity x depth\nfactorial (72)", "#dcefe2"),
             ("Stage 2", "uncertainty x\nmemory (72)\n+ latent delay (30)", "#dcefe2"),
             ("Stages 3-4", "TD3, unstable plant,\ntraining discipline", "#eeeeee"),
             ("Stage 5", "synthesis:\nthese documents", "#dce7f1")]
    gates = ["G0 pass\n(memory headroom)", "G1: architecture,\nnot capacity", "G2 fail, G3 fail", "not run"]
    gcol = ["#2f7a4a", "#2f7a4a", "#b23a24", "#777777"]
    for i, (t, sub, c) in enumerate(boxes):
        x = 0.01 + i * (W + GAP)
        ax.add_patch(plt.Rectangle((x, 0.40), W, 0.50, color=c, ec="#555", lw=0.8, transform=ax.transAxes))
        ax.text(x + 0.01, 0.84, t, transform=ax.transAxes, weight="bold", va="top", fontsize=9.5)
        ax.text(x + 0.01, 0.70, sub, transform=ax.transAxes, va="top", fontsize=8)
        if i < 4:
            xa = x + W
            ax.annotate("", xy=(xa + GAP, 0.65), xytext=(xa, 0.65), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#444"))
            ax.text(xa + GAP / 2, 0.33, gates[i], transform=ax.transAxes, ha="center", va="top", fontsize=7.8,
                    color=gcol[i], weight="bold")
    note = "G4 (practicality): weak pass on the literal rule."
    if lat.get("complete"):
        note += f"   Latent-delay test (G2s): {'H3 survives, narrowed' if lat['survives'] else 'H3 rejection stands'}."
    ax.text(0.01, 0.04, note, transform=ax.transAxes, fontsize=8.3, color="#333")
    save(fig, "f1_stage_map")


# F2 -- Stage 0.1 headroom ------------------------------------------------------
def f2():
    arms = [("A1\nfixed d=1\n+ queue", "fixed1_aug1"), ("A2\nstochastic\n+ queue", "stoch_aug1"),
            ("A3\nstochastic + queue\n+ realised delay", "stoch_aug1_oracle"), ("A4\nstochastic\nno queue", "stoch_aug0")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (lab, c) in enumerate(arms):
        v = [r["final"] for r in runs("s0h", "dnn", c)]
        ax.scatter(i + jitter(len(v), seed=i), v, color=COL["dnn"], s=22, alpha=0.8, zorder=3)
        ax.hlines(st.median(v), i - 0.25, i + 0.25, color="#e76f51", lw=2.2, zorder=4)
    ax.set_xticks(range(4), [a for a, _ in arms], fontsize=8)
    ax.set_ylabel("final avg-100 training return")
    ax.set_title("Stage 0.1: headroom probe (MLP, 6 seeds; bar = median)", fontsize=10)
    save(fig, "f2_headroom")


# F3 -- Stage 1 capacity x depth -------------------------------------------------
def f3():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, field, title in [(axs[0], "final", "training return (avg-100)"), (axs[1], "eval5", "deterministic eval (last 5)")]:
        for fam, ms in [("MLP", ["dnn", "dnn_large"]), ("TRM", ["trm_small", "trm"])]:
            meds = []
            for j, m in enumerate(ms):
                rs = runs("s1", m)
                v = [r["final"] if field == "final" else st.mean([e[1] for e in r["eval"][-5:]]) for r in rs]
                x = np.log10([r["params"] for r in rs])
                ax.scatter(x + jitter(len(v), 0.02, seed=j), v, color=COL[m], s=14, alpha=0.6)
                meds.append((np.log10(st.median(r["params"] for r in rs)), st.median(v)))
            ax.plot(*zip(*meds), color=COL[ms[1]], lw=2, marker="o", label=fam)
        ax.set_xticks([np.log10(135e3), np.log10(788e3)], ["135k", "788k"])
        ax.set_xlabel("actor + critic parameters")
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel("return")
    ax = axs[2]
    for i, m in enumerate(["trm_h1", "trm", "trm_h6"]):
        v = [r["final"] for r in runs("s1", m)]
        ax.scatter(i + jitter(len(v), seed=i), v, color=COL[m], s=14, alpha=0.7)
        ax.hlines(st.median(v), i - 0.25, i + 0.25, color="#222", lw=2)
    ax.axhline(st.median(r["final"] for r in runs("s1", "dnn_large")), color=COL["dnn_large"], ls="--", lw=1.2,
               label="MLP 788k median")
    ax.set_xticks(range(3), ["H=1\n(7 cells/pass)", "H=3\n(21)", "H=6\n(42)"])
    ax.set_title("recursion depth, 788k params (training return)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Stage 1: TRM loses at matched capacity on training return; the final greedy policies nearly tie",
                 fontsize=10, y=1.03)
    save(fig, "f3_capacity_depth")


# F4 -- value fit ----------------------------------------------------------------
def smooth(v, k=9):
    v = np.array([np.nan if x is None else x for x in v], dtype=float)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k // 2): i + k // 2 + 1]
        w = w[~np.isnan(w) & (w > 0)]
        out[i] = np.median(w) if len(w) else np.nan
    return out


def f4():
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.4), gridspec_kw={"wspace": 0.75})
    for m in ["dnn", "dnn_large", "trm_small", "trm", "trm_h1"]:
        curves = np.array([smooth(r["critic"]) for r in runs("s1", m)])
        axs[0].plot(np.nanmedian(curves, axis=0), color=COL[m], lw=1.6, label=LBL[m])
    axs[0].set_yscale("log")
    axs[0].set_xlabel("episode")
    axs[0].set_ylabel("critic loss (median over 12 runs)")
    axs[0].legend(frameon=False, fontsize=7.5)
    axs[0].set_title("value-fit trajectories, Stage 1", fontsize=9.5)
    vf = A["stage1"]["value_fit_ratio"]
    keys = [("trm_small/dnn", "TRM 136k / MLP 135k"), ("trm/dnn_large", "TRM 788k / MLP 788k"),
            ("dnn_large/dnn", "MLP 788k / MLP 135k"), ("trm_h1/trm", "TRM H=1 / H=3")]
    y = np.arange(len(keys))
    for off, ph, c in [(-0.18, "early", "#9bb7d4"), (0.18, "late", "#2b5f8f")]:
        vals = [vf[f"{k}|{ph}"]["median_ratio"] for k, _ in keys]
        axs[1].barh(y + off, vals, height=0.34, color=c, label=f"{ph} (first/last 25%)")
        for yi, k in zip(y, keys):
            s = vf[f"{k[0]}|{ph}"]["sign_ratio_gt1"]
            axs[1].text(vf[f"{k[0]}|{ph}"]["median_ratio"] + 0.03, yi + off, f"{s['positive']}/{s['n']}", va="center", fontsize=7.5)
    axs[1].axvline(1, color="#444", lw=1)
    axs[1].set_yticks(y, [l for _, l in keys], fontsize=8)
    axs[1].set_xlabel("critic-loss ratio (median of 12 paired runs); label = pairs > 1")
    axs[1].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    axs[1].set_xlim(0, 2.2)
    axs[1].set_title("the value-fit deficit is architectural, not capacity", fontsize=9.5)
    save(fig, "f4_value_fit")


# F5 -- Stage 2 interaction --------------------------------------------------------
def f5():
    conds = [("C1\nfixed d=1\n+ queue", "s2", "fixed1_aug1_pa"), ("C3\nstochastic\n+ queue", "s2", "stoch_aug1_pa"),
             ("C4\nstochastic\nno queue", "s2", "stoch_aug0_pa")]
    if A["latent_delay"].get("complete"):
        conds.append(("C6\nlatent per-episode\ndelay + queue", "s2x", "episode_aug1_pa"))
    models = ["dnn", "trm", "trm_carryz", "lstm_carry"]
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    w = 0.19
    for j, m in enumerate(models):
        for i, (_, stg, c) in enumerate(conds):
            v = [r["final"] for r in runs(stg, m, c)]
            if not v:
                continue
            x = i + (j - 1.5) * w
            ax.scatter(x + jitter(len(v), 0.04, seed=i * 10 + j), v, color=COL[m], s=12, alpha=0.6)
            ax.hlines(st.median(v), x - w / 2.4, x + w / 2.4, color=COL[m], lw=2.4, label=LBL[m] if i == 0 else None)
    ax.set_xticks(range(len(conds)), [c[0] for c in conds], fontsize=8)
    ax.set_ylabel("final avg-100 training return")
    ax.set_ylim(-900, -120)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")
    ax.set_title("Stage 2 + latent delay: no trunk that carries memory consistently beats the memoryless MLP "
                 "(y clipped at -900)", fontsize=9.5)
    save(fig, "f5_stage2")


# F6 -- latency ----------------------------------------------------------------
def f6():
    lat = A["cost"]["latency_pendulum_fixed1_aug"]
    order = ["dnn", "dnn_large", "lstm_carry", "trm_h1", "trm_small", "trm", "trm_carryz", "trm_h6"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(order))
    cpu = [lat[f"{m}/cpu1"]["b1_p99_ms"] for m in order]
    gpu = [lat[f"{m}/gpu"]["b1_p99_ms"] for m in order]
    ax.barh(y - 0.18, cpu, height=0.34, color=[COL[m] for m in order], label="1 CPU thread (Xeon Gold 6342)")
    ax.barh(y + 0.18, gpu, height=0.34, color=[COL[m] for m in order], alpha=0.45, hatch="///", label="A100 GPU")
    for hz, ms in [(1000, 1), (500, 2), (100, 10)]:
        ax.axvline(ms, color="#b23a24", ls=":", lw=1)
        ax.text(ms, len(order) - 0.4, f"{hz} Hz", color="#b23a24", fontsize=7.5, ha="center")
    ax.set_xscale("log")
    ax.set_yticks(y, [LBL[m] for m in order], fontsize=8)
    ax.set_xlabel("p99 latency per control decision, batch 1 (ms, log)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("Deployment cost: per-decision latency against control periods", fontsize=9.5)
    save(fig, "f6_latency")


# F7 -- cost ledger --------------------------------------------------------------
def f7():
    c = A["cost"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"wspace": 0.35})
    order = ["dnn", "dnn_large", "trm_h1", "trm_small", "trm", "trm_h6"]
    for dev, mk in [("Tesla V100-SXM2-32GB", "o"), ("NVIDIA A100-PCIE-40GB", "s"), ("NVIDIA L40S", "^")]:
        d = c["per_device_median_min"].get(dev, {})
        xs = [i for i, m in enumerate(order) if m in d]
        axs[0].scatter(xs, [d[order[i]]["median_min"] for i in xs], marker=mk, color="#444", s=26,
                       label=dev.replace("NVIDIA ", "").replace("Tesla ", "").replace("-SXM2-32GB", "").replace("-PCIE-40GB", ""))
    axs[0].set_yscale("log")
    axs[0].set_xticks(range(len(order)), [LBL[m] for m in order], rotation=30, ha="right", fontsize=8)
    axs[0].set_ylabel("median training wall-clock (min, log)")
    axs[0].legend(frameon=False, fontsize=7.5)
    axs[0].set_title("200 episodes of Pendulum-v1, by GPU model", fontsize=9.5)
    mw = c["matched_wallclock_stage1"]
    ms = ["dnn_large", "trm_h1", "trm_small", "trm", "trm_h6"]
    axs[1].bar(range(len(ms)), [mw[m]["median_avg10_at_dnn_finish"] for m in ms], color=[COL[m] for m in ms])
    axs[1].axhline(mw["trm"]["median_dnn_final"], color=COL["dnn"], ls="--", lw=1.4, label="MLP 135k final")
    for i, m in enumerate(ms):
        axs[1].text(i, mw[m]["median_avg10_at_dnn_finish"] - 25, f"episode {int(mw[m]['median_episodes_done_at_dnn_finish'])}",
                    ha="center", va="top", fontsize=7.5, color="#222")
    axs[1].set_xticks(range(len(ms)), [LBL[m] for m in ms], rotation=30, ha="right", fontsize=8)
    axs[1].set_ylabel("return (avg-10) when the MLP finishes")
    axs[1].set_ylim(-1250, 0)
    axs[1].legend(frameon=False, fontsize=7.5)
    axs[1].set_title("matched wall-clock reading", fontsize=9.5)
    save(fig, "f7_cost")


# F8/F9 -- battery2 scorecard and robustness -------------------------------------
def f8_f9():
    b = A["battery2"]
    if not b["n_checkpoints"]:
        print("skip f8/f9: battery2 not ready")
        return
    cells = b["cells"]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    hidden = 0
    for key, c in cells.items():
        stg, m, cond = key.split("|")
        if stg not in ("s1", "s2") or m not in COL:
            continue
        marker = {"none": "o", "fixed1_aug1": "s", "fixed1_aug1_pa": "D", "stoch_aug1_pa": "^", "stoch_aug0_pa": "v"}.get(cond, "x")
        if c["nominal_iqm"] < -170:
            hidden += 1
            continue
        ax.scatter(max(c["command"]["reversals_per_s"], 0.2), c["nominal_iqm"], color=COL[m], marker=marker, s=40,
                   alpha=0.9, ec="white", lw=0.4)
    for m in ["dnn", "dnn_large", "trm_small", "trm", "trm_h1", "trm_h6", "trm_carryz", "lstm_carry"]:
        ax.scatter([], [], color=COL[m], label=LBL[m], s=30)
    for mk, lab in [("o", "no delay"), ("s", "fixed d=1"), ("D", "C1"), ("^", "C3"), ("v", "C4")]:
        ax.scatter([], [], color="#999", marker=mk, label=lab, s=24)
    ax.set_xscale("log")
    ax.set_xlabel("torque direction reversals per second (median over 6 seeds, deterministic, log)")
    ax.set_ylabel("nominal return (IQM, 20 episodes x seeds)")
    ax.legend(frameon=False, fontsize=6.8, ncol=3, loc="lower left")
    ax.set_title("Deployment scorecard, Stages 1-2: TRM trunks cluster at low reversal rates\n"
                 f"at near-equal return ({hidden} cell below -170 not shown: LSTM at C3)", fontsize=9.3)
    save(fig, "f8_scorecard")

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for m in ["dnn", "dnn_large", "trm_small", "trm", "trm_h1", "trm_h6"]:
        c = cells.get(f"s1|{m}|fixed1_aug1")
        if not c:
            continue
        dm = c["delay_mismatch"]
        axs[0].plot([int(k) for k in dm], list(dm.values()), marker="o", ms=3, color=COL[m], label=LBL[m])
        ns = {"0": c["nominal_iqm"], **c["obs_noise"]}
        axs[1].plot([float(k) for k in ns], list(ns.values()), marker="o", ms=3, color=COL[m], label=LBL[m])
    axs[0].axvline(1, color="#999", ls=":")
    axs[0].set_xlabel("test-time delay d (trained at d = 1, one queue slot)")
    axs[0].set_ylabel("return (IQM, median over 6 seeds)")
    axs[0].set_title("delay-mismatch envelope", fontsize=9.5)
    axs[1].set_xlabel("sensor-noise sigma (fraction of observation half-range)")
    axs[1].set_title("sensor-noise envelope", fontsize=9.5)
    axs[1].legend(frameon=False, fontsize=7)
    save(fig, "f9_robustness")


# F10 -- learning curves -----------------------------------------------------------
def f10():
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for ax, cond, title in [(axs[0], "none", "no delay"), (axs[1], "fixed1_aug1", "fixed d = 1 + queue")]:
        for m in ["dnn", "dnn_large", "trm_small", "trm", "trm_h1", "trm_h6"]:
            curves = np.array([np.convolve(r["ret"], np.ones(10) / 10, mode="valid") for r in runs("s1", m, cond)])
            ax.plot(np.median(curves, axis=0), color=COL[m], lw=1.4, label=LBL[m])
        ax.set_ylim(-1500, -100)
        ax.set_xlabel("episode")
        ax.set_title(f"Stage 1 learning curves, {title} (median of 6, 10-ep mean)", fontsize=9)
    axs[0].set_ylabel("training return")
    axs[1].legend(frameon=False, fontsize=7)
    save(fig, "f10_learning_curves")


# F11 -- latent-delay test ------------------------------------------------------------
def f11():
    lat = A["latent_delay"]
    if not lat.get("complete"):
        print("skip f11: latent-delay test not complete")
        return
    arms = [("MLP", "dnn", "episode_aug1_pa", COL["dnn"]), ("MLP +\noracle d", "dnn", "episode_aug1_paoracle", COL["mlp_oracle"]),
            ("TRM\nReset-Z", "trm", "episode_aug1_pa", COL["trm"]), ("TRM\nCarry-Z", "trm_carryz", "episode_aug1_pa", COL["trm_carryz"]),
            ("LSTM\n(carried)", "lstm_carry", "episode_aug1_pa", COL["lstm_carry"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (lab, m, c, col) in enumerate(arms):
        v = [r["final"] for r in runs("s2x", m, c)]
        ax.scatter(i + jitter(len(v), seed=i), v, color=col, s=22, alpha=0.8)
        ax.hlines(st.median(v), i - 0.25, i + 0.25, color="#222", lw=2)
    ax.set_xticks(range(len(arms)), [a[0] for a in arms], fontsize=8)
    ax.set_ylabel("final avg-100 training return")
    ax.set_title("Supplementary: latent per-episode delay d ~ U{0..4} (6 seeds)", fontsize=9.5)
    save(fig, "f11_latent_delay")


if __name__ == "__main__":
    for f in (f1, f2, f3, f4, f5, f6, f7, f8_f9, f10, f11):
        f()
