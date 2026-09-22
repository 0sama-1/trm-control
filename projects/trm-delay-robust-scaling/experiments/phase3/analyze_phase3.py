"""Phase 3 synthesis analysis: everything the write-up quotes, in one JSON.

  python analyze_phase3.py   -> experiments/analysis_phase3_full.json

Inputs: results_phase3/runs_cache.json (extract_runs.py), the battery folders,
and experiments/analysis_w0.json (latency). All contrasts are paired by seed
(and by condition where stated) and use the sign test the gates use.
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "results_phase3"
EXP = PROJ / "experiments"
SEEDS = [1, 2, 3, 4, 5, 6]


# -- stats -------------------------------------------------------------------

def tail(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sign(diffs):
    nz = [d for d in diffs if d != 0 and d == d]
    n, pos = len(nz), sum(1 for d in nz if d > 0)
    if not n:
        return {"n": 0, "positive": 0, "p_greater": None, "p_two": None}
    return {"n": n, "positive": pos, "p_greater": round(tail(pos, n), 4),
            "p_two": round(min(1.0, 2 * min(tail(pos, n), tail(n - pos, n))), 4)}


def paired(x: dict, y: dict, ratio=False):
    keys = sorted(set(x) & set(y), key=str)
    if ratio:
        d = [x[k] / y[k] for k in keys if y[k]]
        s = sign([v - 1 for v in d])
        return {"n": len(d), "median_ratio": round(st.median(d), 3) if d else None, "sign_ratio_gt1": s}
    d = [x[k] - y[k] for k in keys]
    return {"n": len(d), "median": round(st.median(d), 1) if d else None,
            "mean": round(st.mean(d), 1) if d else None, "diffs": [round(v, 1) for v in d], "sign": sign(d)}


def med(v):
    v = [x for x in v if x is not None]
    return round(st.median(v), 1) if v else None


def iqm(v):
    v = sorted(v)
    n = len(v)
    lo, hi = int(math.floor(n * 0.25)), int(math.ceil(n * 0.75))
    return st.mean(v[lo:hi]) if hi > lo else st.mean(v)


def boot_ci(runs, reps=2000, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        pool = []
        for r in (rng.choice(runs) for _ in runs):
            pool += [rng.choice(r) for _ in r]
        out.append(iqm(pool))
    out.sort()
    return [round(out[int(0.025 * reps)], 1), round(out[int(0.975 * reps) - 1], 1)]


# -- run cache helpers ---------------------------------------------------------

RUNS = json.loads((RES / "runs_cache.json").read_text())


def cond_of(r):
    if r["delay"] == "none":
        return "none"
    base = {"fixed": "fixed1", "stochastic": "stoch", "episode": "episode"}[r["delay"]]
    s = f"{base}_aug{int(r['aug'])}"
    if r["tag"]:
        s += "_" + r["tag"]
    return s


def select(stage, model, cond, field="final"):
    return {r["seed"]: r[field] for r in RUNS if r["stage"] == stage and r["model"] == model and cond_of(r) == cond}


def window(v, lo, hi):
    n = len(v)
    xs = [x for x in v[int(n * lo):int(n * hi)] if x]
    return st.median(xs) if xs else None


def episodes_to(ret, thresh=-300.0, k=10):
    for i in range(k, len(ret) + 1):
        if st.mean(ret[i - k:i]) >= thresh:
            return i
    return None


# -- sections ------------------------------------------------------------------

def reproduction():
    """Stage 1 re-runs Phase 1's dnn / trm cells (seeds 1-3, new code, GPU, seeded warmup)."""
    out = {}
    for m in ["dnn", "trm"]:
        for c, pc in [("none", "none"), ("fixed1_aug1", "fixed1_aug1")]:
            p1 = {r["seed"]: r["final"] for r in RUNS if r["stage"] == "p1" and r["model"] == m
                  and ("none" if r["delay"] == "none" else "fixed1_aug1") == pc}
            s1 = select("s1", m, c)
            out[f"{m}|{c}"] = {"phase1_median": med(p1.values()), "stage1_seeds1_3_median": med(s1[s] for s in [1, 2, 3]),
                               "stage1_all6_median": med(s1.values())}
    # the Phase 1 headline (TRM vs the 5.8x smaller MLP), now at n=12
    x, y = {}, {}
    for c in ["none", "fixed1_aug1"]:
        for s, v in select("s1", "trm", c).items():
            x[(c, s)] = v
        for s, v in select("s1", "dnn", c).items():
            y[(c, s)] = v
    out["trm_minus_dnn_n12"] = paired(x, y)
    p1x = {(r["delay"], r["seed"]): r["final"] for r in RUNS if r["stage"] == "p1" and r["model"] == "trm"}
    p1y = {(r["delay"], r["seed"]): r["final"] for r in RUNS if r["stage"] == "p1" and r["model"] == "dnn"}
    out["phase1_trm_minus_dnn_n6"] = paired(p1x, p1y)
    return out


def stage1():
    models = ["dnn", "trm_small", "dnn_large", "trm", "trm_h1", "trm_h6"]
    conds = ["none", "fixed1_aug1"]

    def pooled(model, field="final", fn=None):
        out = {}
        for c in conds:
            for r in RUNS:
                if r["stage"] == "s1" and r["model"] == model and cond_of(r) == c:
                    out[(c, r["seed"])] = fn(r) if fn else r[field]
        return out

    arms = {}
    for m in models:
        rs = [r for r in RUNS if r["stage"] == "s1" and r["model"] == m]
        arms[m] = {
            "params_fixed1_aug1": next((r["params"] for r in rs if cond_of(r) == "fixed1_aug1"), None),
            "median_final": {c: med(r["final"] for r in rs if cond_of(r) == c) for c in conds},
            "median_final_pooled": med(r["final"] for r in rs),
            "median_eval_last5": med(st.mean([e[1] for e in r["eval"][-5:]]) for r in rs),
            "median_wall_min": round(st.median(r["wall_s"] for r in rs) / 60, 1),
            "median_episodes_to_minus300": med(episodes_to(r["ret"]) for r in rs),
            "critic_early_median": med(window(r["critic"], 0.0, 0.25) for r in rs),
            "critic_late_median": med(window(r["critic"], 0.75, 1.0) for r in rs),
        }
    pairs = [("trm_small", "dnn"), ("trm", "dnn_large"), ("dnn_large", "dnn"), ("trm", "trm_small"),
             ("trm_h1", "trm"), ("trm_h6", "trm"), ("trm", "dnn"), ("trm_h1", "dnn")]
    ret = {f"{a}-{b}": paired(pooled(a), pooled(b)) for a, b in pairs}
    ev = {f"{a}-{b}": paired(pooled(a, fn=lambda r: st.mean([e[1] for e in r["eval"][-5:]])),
                             pooled(b, fn=lambda r: st.mean([e[1] for e in r["eval"][-5:]]))) for a, b in pairs}
    vf = {}
    for a, b in pairs:
        for ph, lo, hi in [("early", 0.0, 0.25), ("late", 0.75, 1.0)]:
            vf[f"{a}/{b}|{ph}"] = paired(pooled(a, fn=lambda r: window(r["critic"], lo, hi)),
                                         pooled(b, fn=lambda r: window(r["critic"], lo, hi)), ratio=True)
    speed = {f"{a}-{b}": paired(pooled(a, fn=lambda r: episodes_to(r["ret"]) or 201),
                                pooled(b, fn=lambda r: episodes_to(r["ret"]) or 201)) for a, b in pairs}
    return {"arms": arms, "return_contrasts": ret, "eval_contrasts": ev, "value_fit_ratio": vf,
            "episodes_to_minus300_contrasts": speed}


def cost_ledger(latency):
    """Wall-clock three ways: matched episodes, matched wall-clock (uniform-episode approximation:
    episode k ends at k/200 of the run's wall-clock), matched inference latency."""
    by_dev = defaultdict(lambda: defaultdict(list))
    for r in RUNS:
        if r["stage"] in ("s1", "s2", "s2x", "s0h"):
            by_dev[r["device"]][r["model"]].append(r["wall_s"] / 60)
    per_device = {dev: {m: {"n": len(v), "median_min": round(st.median(v), 1)} for m, v in ms.items()}
                  for dev, ms in by_dev.items()}
    ratios = {}
    for dev, ms in by_dev.items():
        if "dnn" in ms:
            base = st.median(ms["dnn"])
            ratios[dev] = {m: round(st.median(v) / base, 1) for m, v in ms.items()}
    # matched wall-clock on Stage 1: pair each TRM-family run with the dnn run of the same (cond, seed)
    matched = {}
    for m in ["trm_small", "dnn_large", "trm", "trm_h1", "trm_h6"]:
        rows = []
        for c in ["none", "fixed1_aug1"]:
            for s in SEEDS:
                d = next((r for r in RUNS if r["stage"] == "s1" and r["model"] == "dnn" and cond_of(r) == c and r["seed"] == s), None)
                t = next((r for r in RUNS if r["stage"] == "s1" and r["model"] == m and cond_of(r) == c and r["seed"] == s), None)
                if not (d and t):
                    continue
                k = max(1, min(200, int(round(200 * d["wall_s"] / t["wall_s"]))))
                avg10 = st.mean(t["ret"][max(0, k - 10):k])
                rows.append({"episodes_done": k, "avg10_return": round(avg10, 1), "dnn_final": round(d["final"], 1),
                             "same_device": d["device"] == t["device"]})
        matched[m] = {"median_episodes_done_at_dnn_finish": med(r["episodes_done"] for r in rows),
                      "median_avg10_at_dnn_finish": med(r["avg10_return"] for r in rows),
                      "median_dnn_final": med(r["dnn_final"] for r in rows), "n": len(rows)}
    lat = {}
    if latency:
        for k, v in latency["table"].items():
            if k.startswith("pendulum_fixed1_aug/"):
                lat[k.split("/", 1)[1]] = v
    return {"per_device_median_min": per_device, "ratio_to_dnn_same_device": ratios,
            "matched_wallclock_stage1": matched, "latency_pendulum_fixed1_aug": lat}


def stage2():
    conds = {"C1": "fixed1_aug1_pa", "C3": "stoch_aug1_pa", "C4": "stoch_aug0_pa"}
    models = ["dnn", "trm", "trm_carryz", "lstm_carry"]
    v = {m: {c: select("s2", m, cc) for c, cc in conds.items()} for m in models}
    ev = {m: {c: select("s2", m, cc, field="eval") for c, cc in conds.items()} for m in models}
    ev = {m: {c: {s: st.mean([e[1] for e in x[-5:]]) for s, x in d.items()} for c, d in cs.items()} for m, cs in ev.items()}
    out = {"medians": {m: {c: med(d.values()) for c, d in cs.items()} for m, cs in v.items()},
           "median_eval_last5": {m: {c: med(d.values()) for c, d in cs.items()} for m, cs in ev.items()}}

    def gap(a, b, c, src=v):
        return {s: src[a][c][s] - src[b][c][s] for s in set(src[a][c]) & set(src[b][c])}

    for a, b in [("trm_carryz", "dnn"), ("trm_carryz", "trm"), ("trm_carryz", "lstm_carry"), ("lstm_carry", "dnn"), ("trm", "dnn")]:
        for c in conds:
            out[f"{a}-{b}|{c}"] = paired(v[a][c], v[b][c])
        for c in ["C3", "C4"]:
            out[f"interaction[{a}-{b}]|{c}-C1"] = paired(gap(a, b, c), gap(a, b, "C1"))
    for c in ["C3", "C4"]:
        out[f"eval_interaction[trm_carryz-dnn]|{c}-C1"] = paired(gap("trm_carryz", "dnn", c, ev), gap("trm_carryz", "dnn", "C1", ev))
    # previous-action input effect on the MLP (identical code and seeds; Stage 0 arms lack the input)
    s0 = {"C1": "fixed1_aug1", "C3": "stoch_aug1", "C4": "stoch_aug0"}
    out["prev_action_effect_mlp"] = {c: paired(v["dnn"][c], select("s0h", "dnn", s0[c])) for c in conds}
    out["carryz_C4_minus_history_free_mlp_A4"] = paired(v["trm_carryz"]["C4"], select("s0h", "dnn", "stoch_aug0"))
    return out


def latent_delay(s2):
    v = {k: select("s2x", m, c) for k, (m, c) in {
        "mlp": ("dnn", "episode_aug1_pa"), "mlp_oracle": ("dnn", "episode_aug1_paoracle"),
        "trm": ("trm", "episode_aug1_pa"), "carryz": ("trm_carryz", "episode_aug1_pa"),
        "lstm_carry": ("lstm_carry", "episode_aug1_pa")}.items()}
    out = {"n": {k: len(d) for k, d in v.items()}, "medians": {k: med(d.values()) for k, d in v.items()},
           "finals": {k: {str(s): round(x, 1) for s, x in d.items()} for k, d in v.items()}}
    if not all(len(d) == 6 for d in v.values()):
        out["complete"] = False
        return out
    head = paired(v["mlp_oracle"], v["mlp"])
    xs, ys = list(v["mlp_oracle"].values()), list(v["mlp"].values())
    pooled_sd = math.sqrt((st.variance(xs) + st.variance(ys)) / 2)
    head["pooled_sd"] = round(pooled_sd, 1)
    head["passes"] = head["median"] > pooled_sd and head["sign"]["positive"] >= 5
    out["headroom_oracle_minus_mlp"] = head
    out["carryz_minus_mlp"] = paired(v["carryz"], v["mlp"])
    out["carryz_minus_resetz"] = paired(v["carryz"], v["trm"])
    out["lstm_carry_minus_mlp"] = paired(v["lstm_carry"], v["mlp"])
    out["lstm_carry_minus_resetz"] = paired(v["lstm_carry"], v["trm"])
    c1 = select("s2", "trm_carryz", "fixed1_aug1_pa")
    m1 = select("s2", "dnn", "fixed1_aug1_pa")
    g6 = {s: v["carryz"][s] - v["mlp"][s] for s in SEEDS}
    g1 = {s: c1[s] - m1[s] for s in SEEDS}
    out["interaction[carryz-mlp]|C6-C1"] = paired(g6, g1)
    out["survives"] = (head["passes"] and out["carryz_minus_mlp"]["sign"]["positive"] >= 5
                       and out["carryz_minus_resetz"]["sign"]["positive"] >= 5)
    out["complete"] = True
    return out


def battery2():
    # battery2 = all Stage 0.1/1/2 checkpoints; battery3 = the 6 Stage 1 dnn checkpoints whose basenames collide
    # with Stage 0's A1 (so battery2 skipped them); battery4 = the latent-delay checkpoints. Rows are attributed to
    # stages by model_id, never by filename.
    rows, seen = [], set()
    for d in ["results_p3_battery2", "results_p3_battery3", "results_p3_battery4"]:
        for p in sorted((RES / d).glob("*.json")) if (RES / d).exists() else []:
            b = json.loads(p.read_text())
            if b["model_id"] not in seen:
                seen.add(b["model_id"])
                rows.append(b)
    stage_of = {}
    for r in RUNS:
        stage_of[r["model_id"]] = r["stage"]
    cells = defaultdict(list)
    for b in rows:
        stg = stage_of.get(b["model_id"], "?")
        delay = b["delay"]
        c = "none" if delay == "none" else f"{ {'fixed': 'fixed1', 'stochastic': 'stoch', 'episode': 'episode'}[delay] }_aug{int(b['augment_obs'])}"
        if b.get("run_tag"):
            c += "_" + b["run_tag"]
        cells[(stg, b["model"], c)].append(b)
    metrics = ["effort", "abs_du", "abs_d2u", "saturation", "reversals_per_s", "hf_frac", "work", "rms_theta_2nd_half"]
    out = {"n_checkpoints": len(rows), "param_mismatches": sum(1 for b in rows if b["param_count_rebuilt"] != b["param_count_recorded"]),
           "cells": {}}
    for (stg, m, c), bs in sorted(cells.items()):
        runs = [b["probes"]["nominal"]["episodes"]["return"] for b in bs]
        cell = {"n": len(bs), "nominal_iqm": round(iqm([x for r in runs for x in r]), 1), "nominal_ci95": boot_ci(runs),
                "command": {k: round(st.median(b["probes"]["nominal"]["mean"][k] for b in bs), 4)
                            for k in metrics if k in bs[0]["probes"]["nominal"]["mean"]}}
        for probe in ["delay_mismatch", "obs_noise", "plant_mass", "plant_length", "dropout"]:
            cell[probe] = {k: round(st.median(b["probes"][probe][k]["iqm_return"] for b in bs), 1) for k in bs[0]["probes"][probe]}
        out["cells"][f"{stg}|{m}|{c}"] = cell

    # H3.4 / H3.5 on Stage 1 (n = 12 pairs: seed x {none, fixed1_aug1})
    def per_pair(model, fn):
        d = {}
        for (stg, m, c), bs in cells.items():
            if stg == "s1" and m == model:
                for b in bs:
                    d[(c, b["seed"])] = fn(b)
        return d

    nom = lambda b: b["probes"]["nominal"]["iqm_return"]
    du = lambda b: b["probes"]["nominal"]["mean"]["abs_du"]
    rev = lambda b: b["probes"]["nominal"]["mean"]["reversals_per_s"]
    eff = lambda b: b["probes"]["nominal"]["mean"]["effort"]
    mis = lambda b: st.mean(b["probes"]["delay_mismatch"][k]["iqm_return"] for k in ["2", "3"]) - b["probes"]["nominal"]["iqm_return"]
    noi = lambda b: b["probes"]["obs_noise"]["0.05"]["iqm_return"] - b["probes"]["nominal"]["iqm_return"]
    tests = {}
    for a, bm in [("trm", "dnn"), ("trm_small", "dnn"), ("trm", "dnn_large"), ("trm_h1", "trm"), ("dnn_large", "dnn")]:
        tests[f"{a}-{bm}"] = {name: paired(per_pair(a, f), per_pair(bm, f)) for name, f in
                              [("nominal_iqm", nom), ("abs_du", du), ("reversals_per_s", rev), ("effort", eff),
                               ("delay_mismatch_d2_3_drop", mis), ("noise_0.05_drop", noi)]}
    out["stage1_paired"] = tests
    # Stage 2, per condition: Carry-Z vs Reset-Z and vs MLP on command rate and nominal return
    s2t = {}
    for c in ["fixed1_aug1_pa", "stoch_aug1_pa", "stoch_aug0_pa"]:
        def g(model, f):
            return {b["seed"]: f(b) for b in cells.get(("s2", model, c), [])}
        for a, bm in [("trm_carryz", "trm"), ("trm_carryz", "dnn"), ("lstm_carry", "dnn")]:
            s2t[f"{a}-{bm}|{c}"] = {"nominal_iqm": paired(g(a, nom), g(bm, nom)), "abs_du": paired(g(a, du), g(bm, du))}
    out["stage2_paired"] = s2t
    return out


def main():
    w0 = json.loads((EXP / "analysis_w0.json").read_text())
    s2 = stage2()
    out = {
        "reproduction": reproduction(),
        "stage1": stage1(),
        "cost": cost_ledger(w0.get("latency")),
        "stage2": s2,
        "latent_delay": latent_delay(s2),
        "battery2": battery2(),
        "g0": json.loads((EXP / "analysis_p3_g0.json").read_text()),
    }
    (EXP / "analysis_phase3_full.json").write_text(json.dumps(out, indent=1, default=str))
    print("written; battery2 checkpoints:", out["battery2"]["n_checkpoints"], "latent complete:", out["latent_delay"].get("complete"))


if __name__ == "__main__":
    main()
