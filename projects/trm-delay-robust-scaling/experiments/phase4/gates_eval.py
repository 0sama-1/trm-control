"""Phase 4 gate evaluation (rules fixed in gates.json before any data; see phase4/plan/plan.md).

  python gates_eval.py fetch [folders...]   rsync Phase 4 results from Katana into phase4/results/
  python gates_eval.py gb0 [--hopper]       complexity pilot            -> experiments/analysis_p4_b0.json
  python gates_eval.py gc                   LR sweep                    -> experiments/analysis_p4_c.json
  python gates_eval.py a0                   Stage A pilot: clip p99s    -> experiments/analysis_p4_a0.json
  python gates_eval.py ga                   training discipline         -> experiments/analysis_p4_a.json
  python gates_eval.py gt                   G1 under TD3 (Stage T)      -> experiments/analysis_p4_t.json
  python gates_eval.py h1p                  complexity scaling (B1)     -> experiments/analysis_p4_b1.json

A gate command records its outcome in gates.json (status + detail) only when its
data are complete; it never edits a rule, a frozen value, or a gate's "notes" (hand-written
caveats, e.g. GT's optimisation-failure caveat), so rerunning an evaluator is safe.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
RES = PROJ / "phase4" / "results"
RES3 = PROJ / "phase3" / "results"
EXP = PROJ / "experiments"
GATES = HERE / "gates.json"
KATANA = "katana:/srv/scratch/z5308482/trm-control"
FOLDERS = ["results_p4b0", "results_p4b0h", "results_p4b0d", "results_p4t", "results_p4e", "results_p4c", "results_p4a0", "results_p4a", "results_p4b1"]


# ---------------------------------------------------------------------------
# helpers (same statistics as experiments/phase3/gates_eval.py)
# ---------------------------------------------------------------------------

def binom_tail(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sign_test(diffs: list[float]) -> dict:
    nz = [d for d in diffs if d != 0]
    n, pos = len(nz), sum(1 for d in nz if d > 0)
    return {"n": n, "positive": pos, "p_one_sided_greater": binom_tail(pos, n) if n else None,
            "p_two_sided": min(1.0, 2 * min(binom_tail(pos, n), binom_tail(n - pos, n))) if n else None}


def contrast(x: dict, y: dict) -> dict:
    keys = sorted(set(x) & set(y))
    d = [x[k] - y[k] for k in keys]
    xs, ys = [x[k] for k in keys], [y[k] for k in keys]
    pooled = math.sqrt((statistics.variance(xs) + statistics.variance(ys)) / 2) if len(keys) > 1 else None
    return {"pairs": [str(k) for k in keys], "diffs": [round(v, 2) for v in d],
            "median_diff": round(statistics.median(d), 2) if d else None,
            "mean_diff": round(statistics.mean(d), 2) if d else None,
            "pooled_sd": round(pooled, 2) if pooled is not None else None, "sign": sign_test(d)}


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    h = d["episode_history"]
    evals = [r["eval_return"] for r in h if "eval_return" in r]
    cl = [r["critic_loss"] for r in h if "critic_loss" in r]
    late = cl[int(0.75 * len(cl)):]
    return {"final": d["final_avg100_return"], "eval_last5": statistics.mean(evals[-5:]) if evals else None,
            "critic_late": statistics.median(late) if late else None, "wall_s": d["wall_clock_seconds"],
            "steps": d.get("total_steps", h[-1]["total_steps"] if h else None), "episodes": d["episode_count"],
            "params": d["param_count"], "device": d.get("device_name"), "history": h}


def gates() -> dict:
    return json.loads(GATES.read_text())


def set_gate(gid: str, status: str, detail: str) -> None:
    g = gates()
    for x in g["gates"]:
        if x["id"] == gid:
            x["status"], x["detail"] = status, detail
    GATES.write_text(json.dumps(g, indent=1))


def avg100_at(history: list[dict], budget: int) -> float | None:
    """avg-100 after the last episode that started before `budget` env steps (the step-budget rule)."""
    prev_steps, val = 0, None
    for row in history:
        if prev_steps >= budget:
            break
        val, prev_steps = row["avg100_return"], row["total_steps"]
    return val


def first_cross(history: list[dict], thr: float) -> int | None:
    return next((r["total_steps"] for r in history if r["avg100_return"] >= thr and r["episode"] >= 100), None)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def fetch(folders=None) -> None:
    RES.mkdir(exist_ok=True)
    for f in folders or FOLDERS:
        subprocess.run(["rsync", "-a", "--ignore-missing-args", f"{KATANA}/{f}/", str(RES / f) + "/"], check=False)
        n = len(list((RES / f).glob("*.json"))) if (RES / f).exists() else 0
        print(f"{f}: {n} files")


def gb0(hopper: bool = False) -> None:
    fv = gates()["frozen_values"]
    ref = fv["b0h_reference" if hopper else "b0_reference"]
    env_id, thr = ref["env"], ref["learned_threshold"]
    folder = RES / ("results_p4b0h" if hopper else "results_p4b0")
    budgets = fv["b0_budgets_steps"]
    out = {"env": env_id, "threshold": thr, "budgets": {}, "timing": {}}
    for b in budgets:
        runs = {s: load(folder / f"{env_id}_td3_dnn_fixed_aug1_{b // 1000}k_seed{s}.json") for s in [1, 2, 3]}
        runs = {s: r for s, r in runs.items() if r}
        out["budgets"][str(b)] = {
            "n": len(runs),
            "finals": {str(s): round(r["final"], 1) for s, r in runs.items()},
            "learned": {str(s): r["final"] >= thr for s, r in runs.items()},
            "n_learned": sum(r["final"] >= thr for r in runs.values()),
            "eval_last5": {str(s): round(r["eval_last5"], 1) for s, r in runs.items() if r["eval_last5"] is not None},
            "steps_to_threshold": {str(s): first_cross(r["history"], thr) for s, r in runs.items()},
            "wall_min": {str(s): round(r["wall_s"] / 60, 1) for s, r in runs.items()},
            "devices": sorted({r["device"] for r in runs.values() if r["device"]}),
        }
    # Nested-budget reproducibility check: the 500k runs' avg-100 at each smaller budget.
    big = budgets[-1]
    for s in [1, 2, 3]:
        r = load(folder / f"{env_id}_td3_dnn_fixed_aug1_{big // 1000}k_seed{s}.json")
        if r:
            out.setdefault("nested_check", {})[str(s)] = {str(b): avg100_at(r["history"], b) for b in budgets[:-1]}
    if not hopper:
        t = budgets[0] // 10
        for m in ["trm", "trm_h1"]:
            r = load(folder / f"{env_id}_td3_{m}_fixed_aug1_timing{t // 1000}k_seed1.json")
            if r:
                per_step = r["wall_s"] / r["steps"]
                out["timing"][m] = {"steps": r["steps"], "wall_s": round(r["wall_s"]), "device": r["device"],
                                    "s_per_step": round(per_step, 5),
                                    "projected_h": {str(b): round(per_step * b / 3600, 2) for b in budgets}}
        mlp_ps = [r["wall_s"] / r["steps"] for b in budgets for s in [1, 2, 3]
                  if (r := load(folder / f"{env_id}_td3_dnn_fixed_aug1_{b // 1000}k_seed{s}.json"))]
        if mlp_ps:
            out["timing"]["dnn"] = {"s_per_step_median": round(statistics.median(mlp_ps), 5)}
    complete = all(v["n"] == 3 for v in out["budgets"].values())
    passing = [int(b) for b, v in out["budgets"].items() if v["n_learned"] >= 2]
    out["complete"], out["passing_budgets"] = complete, passing
    out["chosen_budget"] = min(passing) if passing else None
    gid = "KB0" if hopper else "GB0"
    out[gid] = ("pass" if passing else "fail") if complete else "pending"
    name = "analysis_p4_b0h.json" if hopper else "analysis_p4_b0.json"
    (EXP / name).write_text(json.dumps(out, indent=1))
    detail = "; ".join(f"{int(b) // 1000}k: {v['n_learned']}/{v['n']} learned (finals {list(v['finals'].values())})"
                       for b, v in out["budgets"].items())
    detail += f". Chosen budget: {out['chosen_budget']}." if passing else ". No budget passes."
    if complete:
        set_gate(gid, out[gid], detail)
    print(out[gid], detail)
    print(json.dumps(out["timing"], indent=1))


def gc() -> None:
    fv = gates()["frozen_values"]["stage_c_lr"]
    mults = ["0.3", "1", "3"]
    seeds = fv["seeds"]
    cells = {}
    for a in mults:
        for c in mults:
            runs = {s: load(RES / "results_p4c" / f"Pendulum-v1_ddpg_trm_fixed_aug1_lra{a}c{c}_seed{s}.json")
                    for s in seeds}
            cells[f"a{a}c{c}"] = {s: r for s, r in runs.items() if r}
    base = cells["a1c1"]
    out = {"cells": {}, "flagged": []}
    allf = [r["final"] for rs in cells.values() for r in rs.values()]
    for k, rs in cells.items():
        row = {"n": len(rs), "median_final": med(r["final"] for r in rs.values()),
               "finals": {str(s): round(r["final"], 1) for s, r in rs.items()},
               "critic_late_median": med(r["critic_late"] for r in rs.values()),
               "eval_last5_median": med(r["eval_last5"] for r in rs.values()),
               "wall_min_median": med(r["wall_s"] / 60 for r in rs.values())}
        if k != "a1c1":
            ct = contrast({s: r["final"] for s, r in rs.items()}, {s: r["final"] for s, r in base.items()})
            row["vs_default"] = ct
            row["flag"] = (ct["sign"]["n"] == 3 and ct["pooled_sd"] is not None
                           and all(d > ct["pooled_sd"] for d in ct["diffs"]))
            if row["flag"]:
                out["flagged"].append(k)
        out["cells"][k] = row
    out["pooled_sd_all_cells"] = round(statistics.pstdev(allf), 2) if allf else None
    complete = all(len(rs) == 3 for rs in cells.values())
    out["complete"] = complete
    out["GC"] = ("flag" if out["flagged"] else "insensitive") if complete else "pending"
    (EXP / "analysis_p4_c.json").write_text(json.dumps(out, indent=1))
    detail = ("Flagged: " + ", ".join(out["flagged"]) if out["flagged"]
              else "No cell beats x1/x1 by more than the pooled SD in 3/3 seeds: TRM insensitive to LR within x0.3-x3") + \
             ". Medians: " + ", ".join(f"{k} {v['median_final']}" for k, v in out["cells"].items())
    if complete:
        set_gate("GC", out["GC"], detail)
    print(out["GC"], detail)


def a0() -> None:
    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)]

    out = {"runs": {}}
    pool = {"actor": [], "critic": []}
    for m, c, t in [("trm", "fixed_aug1", "C1"), ("trm", "stochastic_aug0", "C4"),
                    ("trm_fullgrad", "fixed_aug1", "C1"), ("trm_fullgrad_ds", "fixed_aug1", "C1")]:
        r = load(RES / "results_p4a0" / f"Pendulum-v1_ddpg_{m}_{c}_pa_norms_seed1.json")
        if not r:
            continue
        norms = {k: [x for row in r["history"] for x in row.get(f"{k}_grad_norms", [])] for k in pool}
        out["runs"][f"{m}|{t}"] = {
            "final": round(r["final"], 1), "wall_min": round(r["wall_s"] / 60, 1), "device": r["device"],
            **{f"{k}_p50": round(pct(v, 0.5), 4) for k, v in norms.items() if v},
            **{f"{k}_p99": round(pct(v, 0.99), 4) for k, v in norms.items() if v},
            **{f"{k}_max": round(max(v), 4) for k, v in norms.items() if v}}
        if m == "trm":
            for k in pool:
                pool[k] += norms[k]
    sig2 = lambda x: float(f"{x:.2g}")
    if pool["actor"] and pool["critic"]:
        out["proposed_clip"] = {k: sig2(pct(v, 0.99)) for k, v in pool.items()}
        for key, run in out["runs"].items():
            m = key.split("|")[0]
            r = load(RES / "results_p4a0" / f"Pendulum-v1_ddpg_{m}_{'stochastic_aug0' if key.endswith('C4') else 'fixed_aug1'}_pa_norms_seed1.json")
            for k in pool:
                v = [x for row in r["history"] for x in row.get(f"{k}_grad_norms", [])]
                run[f"{k}_would_clip_frac"] = round(sum(x > out["proposed_clip"][k] for x in v) / len(v), 4) if v else None
    (EXP / "analysis_p4_a0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def ga() -> None:
    reuse = gates()["frozen_values"].get("stage_a_reuse_mlp")
    conds = {"C1": "fixed_aug1", "C4": "stochastic_aug0"}
    arms = {"trm": "pa_clip", "trm_fullgrad": "pa_clip", "trm_fullgrad_ds": "pa_clip"}
    v, meta = {}, {}
    for m, tag in arms.items():
        for c, ct in conds.items():
            rs = {s: load(RES / "results_p4a" / f"Pendulum-v1_ddpg_{m}_{ct}_{tag}_seed{s}.json") for s in range(1, 7)}
            v[(m, c)] = {s: r for s, r in rs.items() if r}
    for c, ct in conds.items():
        folder = RES3 / "results_p3s2" if reuse else RES / "results_p4a"
        rs = {s: load(folder / f"Pendulum-v1_ddpg_dnn_{ct}_pa_seed{s}.json") for s in range(1, 7)}
        v[("dnn", c)] = {s: r for s, r in rs.items() if r}
    f = lambda m, c: {s: r["final"] for s, r in v[(m, c)].items()}
    out = {"mlp_source": "results_p3s2 (reuse, amendment R-1)" if reuse else "results_p4a", "arms": {}, "contrasts": {}}
    for (m, c), rs in v.items():
        out["arms"][f"{m}|{c}"] = {"n": len(rs), "median_final": med(r["final"] for r in rs.values()),
                                   "eval_last5_median": med(r["eval_last5"] for r in rs.values()),
                                   "critic_late_median": med(r["critic_late"] for r in rs.values()),
                                   "wall_min_median": med(r["wall_s"] / 60 for r in rs.values()),
                                   "devices": sorted({r["device"] for r in rs.values() if r["device"]})}
    passing = []
    for d in ["trm_fullgrad", "trm_fullgrad_ds"]:
        for c in conds:
            ct = contrast(f(d, c), f("trm", c))
            out["contrasts"][f"{d} - trm | {c}"] = ct
            if ct["sign"]["n"] == 6 and ct["sign"]["positive"] >= 5:
                passing.append(f"{d} - trm | {c}")
            out["contrasts"][f"{d} - dnn | {c} (secondary)"] = contrast(f(d, c), f("dnn", c))
            for base in ["trm", "dnn"]:
                a = {s: r["critic_late"] for s, r in v[(d, c)].items()}
                b = {s: r["critic_late"] for s, r in v[(base, c)].items()}
                ks = sorted(set(a) & set(b))
                ratios = [a[k] / b[k] for k in ks if b[k]]
                out["contrasts"][f"critic-late ratio {d}/{base} | {c}"] = {
                    "median_ratio": med(ratios), "n_above_1": sum(x > 1 for x in ratios), "n": len(ratios)}
    complete = all(len(rs) == 6 for rs in v.values())
    out["complete"], out["passing"] = complete, passing
    out["GA"] = ("pass" if passing else "fail") if complete else "pending"
    (EXP / "analysis_p4_a.json").write_text(json.dumps(out, indent=1))
    detail = "; ".join(f"{k}: median {c['median_diff']:+.1f}, {c['sign']['positive']}/{c['sign']['n']}"
                       for k, c in out["contrasts"].items() if "ratio" not in k and "secondary" not in k)
    if complete:
        set_gate("GA", out["GA"], detail + (f". Passing: {', '.join(passing)}." if passing else ". No contrast reaches 5/6."))
    print(out["GA"], detail)


def b0d() -> None:
    """IDP diagnostic (amendment K-2): C1 at 1M steps and no delay at 500k, MLP, TD3, seeds 1-3.
    Reported against the 8190 bar with learning curves; not a gate."""
    thr = gates()["frozen_values"]["b0_reference"]["learned_threshold"]
    arms = {"C1_1000k": ("fixed_aug1_1000k", 1_000_000), "C0_500k": ("none_500k", 500_000)}
    out = {"threshold": thr, "arms": {}}
    for arm, (tag, budget) in arms.items():
        rows = {}
        for s in [1, 2, 3]:
            r = load(RES / "results_p4b0d" / f"InvertedDoublePendulum-v5_td3_dnn_{tag}_seed{s}.json")
            if not r:
                continue
            h = r["history"]
            curve = []
            for lo in range(0, budget, 100_000):
                eps = [x for x in h if lo < x["total_steps"] <= lo + 100_000]
                curve.append({"to_k": (lo + 100_000) // 1000, "episodes": len(eps),
                              "avg100_end": round(eps[-1]["avg100_return"], 1) if eps else None,
                              "pct_solved": round(100 * sum(x["return"] > 9000 for x in eps) / len(eps), 1) if eps else None})
            ev = [x["eval_return"] for x in h if "eval_return" in x][-30:]
            rows[str(s)] = {"final": round(r["final"], 1), "learned": r["final"] >= thr,
                            "first_cross": first_cross(h, thr), "best_avg100": round(max(x["avg100_return"] for x in h), 1),
                            "last30_evals_solved": sum(x > 9000 for x in ev), "curve": curve,
                            "wall_min": round(r["wall_s"] / 60, 1), "device": r["device"]}
        out["arms"][arm] = {"seeds": rows, "n_learned": sum(v["learned"] for v in rows.values()), "n": len(rows)}
    c1 = out["arms"]["C1_1000k"]
    out["passes_gb0_rule_at_1M"] = c1["n"] == 3 and c1["n_learned"] >= 2
    (EXP / "analysis_p4_b0d.json").write_text(json.dumps(out, indent=1))
    for arm, v in out["arms"].items():
        print(arm, f"{v['n_learned']}/{v['n']} learned")
        for s, x in v["seeds"].items():
            print(f"  seed {s}: final {x['final']}, best avg100 {x['best_avg100']}, first >= {thr} at {x['first_cross']}, "
                  f"last-30 evals solved {x['last30_evals_solved']}/30, {x['wall_min']} min {x['device']}")
            print("    " + " | ".join(f"{c['to_k']}k: {c['avg100_end']} ({c['pct_solved']}%)" for c in x["curve"]))
    print("passes GB0 rule at 1M:", out["passes_gb0_rule_at_1M"])


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def e() -> None:
    """Stage E (exploratory): actor Lipschitz vs Phase 3 battery reversals/s and sensor-noise drop."""
    lip = {}
    for p in sorted((RES / "results_p4e").glob("*.json")):
        d = json.loads(p.read_text())
        lip[d["model_id"]] = d
    bat = {}
    for folder in ["results_p3_battery2", "results_p3_battery3"]:
        for p in sorted((RES3 / folder).glob("*.json")):
            d = json.loads(p.read_text())
            if d["model_id"] in lip:
                bat[d["model_id"]] = d
    rows = []
    for mid, l in lip.items():
        b = bat.get(mid)
        if not b:
            continue
        nom = b["probes"]["nominal"]
        rows.append({"model": l["model"], "seed": l["seed"], "delay": l["delay"], "lip_median": l["lip_plant"]["median"],
                     "lip_p95": l["lip_plant"]["p95"], "reversals_per_s": nom["mean"]["reversals_per_s"],
                     "abs_du": nom["mean"]["abs_du"],
                     "noise_drop": nom["iqm_return"] - b["probes"]["obs_noise"]["0.05"]["iqm_return"],
                     "nominal_iqm": nom["iqm_return"]})
    fam = lambda m: "MLP" if m.startswith("dnn") else "TRM"
    out = {"label": "exploratory", "n_checkpoints": len(lip), "n_joined": len(rows), "per_model": {}, "spearman": {}}
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]
        out["per_model"][m] = {"n": len(rs), **{k: med(r[k] for r in rs) for k in
                                                ["lip_median", "lip_p95", "reversals_per_s", "noise_drop", "nominal_iqm"]}}
    for scope, rs in [("all", rows), ("MLP", [r for r in rows if fam(r["model"]) == "MLP"]),
                      ("TRM", [r for r in rows if fam(r["model"]) == "TRM"])]:
        out["spearman"][scope] = {"n": len(rs),
                                  "lip_vs_reversals": spearman([r["lip_median"] for r in rs], [r["reversals_per_s"] for r in rs]),
                                  "lip_vs_noise_drop": spearman([r["lip_median"] for r in rs], [r["noise_drop"] for r in rs])}
    out["rows"] = rows
    (EXP / "analysis_p4_e.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


def gt() -> None:
    """Stage T: Phase 3 Stage 1's G1 contrasts under TD3 (plan §9b)."""
    models = ["dnn", "dnn_large", "trm_small", "trm"]
    conds = {"none": "none", "fixed1": "fixed_aug1"}
    runs = {m: {(c, s): load(RES / "results_p4t" / f"Pendulum-v1_td3_{m}_{t}_seed{s}.json")
                for c, t in conds.items() for s in range(1, 7)} for m in models}
    runs = {m: {k: r for k, r in rs.items() if r} for m, rs in runs.items()}
    fin = {m: {k: r["final"] for k, r in rs.items()} for m, rs in runs.items()}
    p3 = {m: {(c, s): (load(RES3 / "results_p3s1" / f"Pendulum-v1_ddpg_{m}_{t}_seed{s}.json") or {}).get("final")
              for c, t in conds.items() for s in range(1, 7)} for m in models}
    out = {"arms": {}, "contrasts": {}, "ddpg_stage1_medians": {m: med(v.values()) for m, v in p3.items()}}
    for m, rs in runs.items():
        out["arms"][m] = {"n": len(rs), "median_final": med(r["final"] for r in rs.values()),
                          "eval_last5_median": med(r["eval_last5"] for r in rs.values()),
                          "critic_late_median": med(r["critic_late"] for r in rs.values()),
                          "wall_min_median": med(r["wall_s"] / 60 for r in rs.values()),
                          "params": next(iter(rs.values()))["params"] if rs else None}
    for label, (a, b) in {"trm - dnn_large (GT)": ("trm", "dnn_large"), "trm_small - dnn": ("trm_small", "dnn"),
                          "dnn_large - dnn": ("dnn_large", "dnn"), "trm - trm_small": ("trm", "trm_small")}.items():
        c = contrast(fin[a], fin[b])
        c["different"] = c["sign"]["n"] == 12 and max(c["sign"]["positive"], 12 - c["sign"]["positive"]) >= 10
        out["contrasts"][label] = c
    for a, b in [("trm", "dnn_large"), ("trm_small", "dnn")]:
        ks = sorted(set(runs[a]) & set(runs[b]))
        ratios = [runs[a][k]["critic_late"] / runs[b][k]["critic_late"] for k in ks if runs[b][k]["critic_late"]]
        out["contrasts"][f"critic-late ratio {a}/{b}"] = {"median_ratio": med(ratios),
                                                          "n_above_1": sum(x > 1 for x in ratios), "n": len(ratios)}
    g = out["contrasts"]["trm - dnn_large (GT)"]
    complete = all(len(v) == 12 for v in fin.values())
    out["complete"] = complete
    out["GT"] = ("pass" if g["sign"]["n"] == 12 and g["sign"]["n"] - g["sign"]["positive"] >= 10 else "fail") if complete else "pending"
    (EXP / "analysis_p4_t.json").write_text(json.dumps(out, indent=1, default=str))
    detail = (f"trm - dnn_large under TD3: median {g['median_diff']}, negative in {g['sign']['n'] - g['sign']['positive']}/"
              f"{g['sign']['n']} pairs (two-sided p={g['sign']['p_two_sided']}). " +
              "; ".join(f"{k}: {c['median_diff']} ({c['sign']['positive']}/{c['sign']['n']} positive)"
                        for k, c in out["contrasts"].items() if "ratio" not in k and "GT" not in k))
    if complete:
        set_gate("GT", out["GT"], detail)
    print(out["GT"], detail)


def h1p() -> None:
    fv = gates()["frozen_values"]
    b1 = fv["b1"]
    env_new = b1["env_id"]
    untrained = fv["span"]["untrained"]
    pend = {m: {s: load(RES3 / "results_p3s1" / f"Pendulum-v1_ddpg_{m}_fixed_aug1_seed{s}.json") for s in range(1, 7)}
            for m in ["dnn", "trm_small", "dnn_large", "trm", "trm_h1", "trm_h6"]}
    new = {m: {s: load(RES / "results_p4b1" / f"{env_new}_td3_{m}_fixed_aug1_seed{s}.json") for s in range(1, 7)}
           for m in ["dnn", "dnn_large", "trm", "trm_h1"]}
    fin = lambda d: {m: {s: r["final"] for s, r in rs.items() if r} for m, rs in d.items()}
    P, N = fin(pend), fin(new)
    span_p = max(x for rs in P.values() for x in rs.values()) - untrained["Pendulum-v1"]
    span_n = max(x for rs in N.values() for x in rs.values()) - untrained[env_new] if any(N.values()) else None
    out = {"span": {"Pendulum-v1": round(span_p, 1), env_new: round(span_n, 1) if span_n else None}, "tests": {}}
    for label, (mlp, trm) in {"primary: dnn vs trm": ("dnn", "trm"), "dnn_large vs trm": ("dnn_large", "trm"),
                              "dnn vs trm_h1": ("dnn", "trm_h1")}.items():
        dp = {s: (P[mlp][s] - P[trm][s]) / span_p for s in set(P[mlp]) & set(P[trm])}
        dn = {s: (N[mlp][s] - N[trm][s]) / span_n for s in set(N[mlp]) & set(N[trm])} if span_n else {}
        ct = contrast(dp, dn)   # positive = deficit smaller on the new rung
        out["tests"][label] = {"delta_pendulum": {str(k): round(x, 4) for k, x in dp.items()},
                               "delta_new": {str(k): round(x, 4) for k, x in dn.items()},
                               "pendulum_minus_new": ct,
                               "supported": ct["sign"]["n"] == 6 and ct["sign"]["positive"] >= 5}
    complete = all(len(rs) == 6 for rs in N.values())
    out["complete"] = complete
    prim = out["tests"]["primary: dnn vs trm"]
    out["H1p"] = ("supported" if prim["supported"] else "not supported") if complete else "pending"
    out["arms_new"] = {m: {"median": med(rs.values()), "n": len(rs)} for m, rs in N.items()}
    (EXP / "analysis_p4_b1.json").write_text(json.dumps(out, indent=1))
    c = prim["pendulum_minus_new"]
    detail = (f"Normalised deficit smaller on {env_new} in {c['sign']['positive']}/{c['sign']['n']} seed pairs "
              f"(median delta_P - delta_new {c['median_diff']}); spans {out['span']}.")
    if complete:
        set_gate("H1p", out["H1p"], detail)
    print(out["H1p"], detail)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "fetch":
        fetch(args or None)
    elif cmd == "gb0":
        gb0(hopper="--hopper" in args)
    else:
        {"gc": gc, "a0": a0, "ga": ga, "gt": gt, "b0d": b0d, "e": e, "h1p": h1p}[cmd]()
