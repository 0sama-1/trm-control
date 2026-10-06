"""Phase 3 gate evaluation (rules fixed in dashboard_state/gates.json before any data).

  python gates_eval.py fetch              rsync Phase 3 results from Katana into phase3/results/
  python gates_eval.py g0                 headroom gate from Stage 0.1   -> experiments/analysis_p3_g0.json
  python gates_eval.py g1                 attribution gate from Stage 1  -> experiments/analysis_s1.json
  python gates_eval.py w0                 battery + latency + G4         -> experiments/analysis_w0.json
  python gates_eval.py g2                 interaction gate from Stage 2  -> experiments/analysis_s2.json

Each gate command also writes its decision into dashboard_state/gates.json.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
RES = PROJ / "phase3" / "results"
EXP = PROJ / "experiments"
GATES = HERE / "dashboard_state" / "gates.json"
KATANA = "katana:/srv/scratch/z5308482/trm-control"
SEEDS = [1, 2, 3, 4, 5, 6]
FOLDERS = ["results_p3s0", "results_p3s1", "results_p3s2", "results_p3s0_battery", "results_p3s0_latency"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def binom_tail(k: int, n: int) -> float:
    """P(X >= k), X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sign_test(diffs: list[float]) -> dict:
    nz = [d for d in diffs if d != 0]
    n, pos = len(nz), sum(1 for d in nz if d > 0)
    return {"n": n, "positive": pos, "p_one_sided_greater": binom_tail(pos, n) if n else None,
            "p_two_sided": min(1.0, 2 * min(binom_tail(pos, n), binom_tail(n - pos, n))) if n else None}


def load_scalar(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    evals = [r["eval_return"] for r in d["episode_history"] if "eval_return" in r]
    return {"final": d["final_avg100_return"], "eval_last5": statistics.mean(evals[-5:]) if evals else None,
            "wall_s": d["wall_clock_seconds"], "params": d["param_count"], "device": d.get("device_name")}


def contrast(x: dict[int, float], y: dict[int, float]) -> dict:
    keys = sorted(set(x) & set(y))
    d = [x[k] - y[k] for k in keys]
    xs, ys = [x[k] for k in keys], [y[k] for k in keys]
    pooled = math.sqrt((statistics.variance(xs) + statistics.variance(ys)) / 2) if len(keys) > 1 else None
    return {"pairs": [str(k) for k in keys], "diffs": [round(v, 2) for v in d],
            "median_diff": round(statistics.median(d), 2) if d else None,
            "mean_diff": round(statistics.mean(d), 2) if d else None,
            "pooled_sd": round(pooled, 2) if pooled is not None else None, "sign": sign_test(d)}


def set_gate(gid: str, status: str, detail: str) -> None:
    g = json.loads(GATES.read_text())
    for x in g["gates"]:
        if x["id"] == gid:
            x["status"], x["detail"] = status, detail
    GATES.write_text(json.dumps(g, indent=1))


def iqm(v: list[float]) -> float:
    v = sorted(v)
    n = len(v)
    lo, hi = int(math.floor(n * 0.25)), int(math.ceil(n * 0.75))
    return statistics.mean(v[lo:hi]) if hi > lo else statistics.mean(v)


def boot_iqm(runs: list[list[float]], reps: int = 2000, seed: int = 0) -> list[float]:
    """Percentile CI of the IQM of pooled episodes, resampling runs then episodes."""
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        pool = []
        for r in (rng.choice(runs) for _ in runs):
            pool += [rng.choice(r) for _ in r]
        out.append(iqm(pool))
    out.sort()
    return [round(out[int(0.025 * reps)], 1), round(out[int(0.975 * reps) - 1], 1)]


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def fetch() -> None:
    RES.mkdir(exist_ok=True)
    for f in FOLDERS:
        subprocess.run(["rsync", "-a", "--ignore-missing-args", f"{KATANA}/{f}/", str(RES / f) + "/"], check=False)
        n = len(list((RES / f).glob("*.json"))) if (RES / f).exists() else 0
        print(f"{f}: {n} files")


def g0() -> None:
    base = RES / "results_p3s0"
    arms = {
        "A1_fixed1_aug": "Pendulum-v1_ddpg_dnn_fixed_aug1_seed{s}.json",
        "A2_stoch_aug": "Pendulum-v1_ddpg_dnn_stochastic_aug1_seed{s}.json",
        "A3_stoch_oracle": "Pendulum-v1_ddpg_dnn_stochastic_aug1_oracle_seed{s}.json",
        "A4_stoch_noaug": "Pendulum-v1_ddpg_dnn_stochastic_aug0_seed{s}.json",
    }
    data = {a: {s: load_scalar(base / f.format(s=s)) for s in SEEDS} for a, f in arms.items()}
    finals = {a: {s: v["final"] for s, v in d.items() if v} for a, d in data.items()}
    evals = {a: {s: v["eval_last5"] for s, v in d.items() if v and v["eval_last5"] is not None} for a, d in data.items()}
    out = {"arms": {a: {"finals": {str(k): round(v, 1) for k, v in f.items()},
                        "median": round(statistics.median(f.values()), 1) if f else None} for a, f in finals.items()}}

    def judge(c):
        return (c["sign"]["n"] == 6 and c["median_diff"] is not None and c["pooled_sd"] is not None
                and c["median_diff"] > c["pooled_sd"] and c["sign"]["positive"] >= 5)

    out["oracle_headroom_A3_minus_A2"] = c_o = contrast(finals["A3_stoch_oracle"], finals["A2_stoch_aug"])
    out["memory_headroom_A2_minus_A4"] = c_m = contrast(finals["A2_stoch_aug"], finals["A4_stoch_noaug"])
    out["stochastic_cost_A1_minus_A2"] = contrast(finals["A1_fixed1_aug"], finals["A2_stoch_aug"])
    out["secondary_eval_last5"] = {
        "oracle": contrast(evals["A3_stoch_oracle"], evals["A2_stoch_aug"]),
        "memory": contrast(evals["A2_stoch_aug"], evals["A4_stoch_noaug"]),
    }
    complete = all(len(f) == 6 for f in finals.values())
    out["oracle_passes"], out["memory_passes"] = judge(c_o), judge(c_m)
    out["complete"] = complete
    out["G0"] = ("pass" if (out["oracle_passes"] or out["memory_passes"]) else "fail") if complete else "pending"
    (EXP / "analysis_p3_g0.json").write_text(json.dumps(out, indent=1))
    fmt = lambda c: (f"median {c['median_diff']:+.1f} vs pooled SD {c['pooled_sd']:.1f}, "
                     f"{c['sign']['positive']}/{c['sign']['n']} seeds positive (p={c['sign']['p_one_sided_greater']:.3f})")
    detail = (f"Oracle headroom A3-A2: {fmt(c_o)} -> {'passes' if out['oracle_passes'] else 'fails'}. "
              f"Memory headroom A2-A4: {fmt(c_m)} -> {'passes' if out['memory_passes'] else 'fails'}.")
    if complete:
        set_gate("G0", out["G0"], detail)
    print(out["G0"], detail)


def g1() -> None:
    base = RES / "results_p3s1"
    models = ["dnn", "trm_small", "dnn_large", "trm", "trm_h1", "trm_h6"]
    conds = {"none": "none", "fixed1": "fixed_aug1"}
    vals, meta = {}, {}
    for m in models:
        vals[m] = {}
        for c, tag in conds.items():
            for s in SEEDS:
                v = load_scalar(base / f"Pendulum-v1_ddpg_{m}_{tag}_seed{s}.json")
                if v:
                    vals[m][(c, s)] = v["final"]
                    meta.setdefault(m, []).append(v)
    pairs = {
        "small: trm_small - dnn": ("trm_small", "dnn"),
        "large: trm - dnn_large": ("trm", "dnn_large"),
        "MLP capacity: dnn_large - dnn": ("dnn_large", "dnn"),
        "TRM capacity: trm - trm_small": ("trm", "trm_small"),
        "depth: trm_h1 - trm": ("trm_h1", "trm"),
        "depth: trm_h6 - trm": ("trm_h6", "trm"),
    }
    out = {"contrasts": {}, "arms": {}}
    for label, (a, b) in pairs.items():
        c = contrast(vals[a], vals[b])
        c["pairs"] = [f"{k}" for k in c["pairs"]]
        c["different"] = c["sign"]["n"] == 12 and max(c["sign"]["positive"], 12 - c["sign"]["positive"]) >= 10
        out["contrasts"][label] = c
    for m in models:
        f = list(vals[m].values())
        ms = meta.get(m, [])
        out["arms"][m] = {"n": len(f), "median": round(statistics.median(f), 1) if f else None,
                          "params": ms[0]["params"] if ms else None,
                          "median_wall_min": round(statistics.median(x["wall_s"] for x in ms) / 60, 1) if ms else None,
                          "devices": sorted({x["device"] for x in ms if x["device"]})}
    med = {m: out["arms"][m]["median"] for m in models}
    if all(v is not None for v in med.values()):
        out["interaction_arch_x_capacity"] = round((med["trm"] - med["dnn_large"]) - (med["trm_small"] - med["dnn"]), 1)
    complete = all(out["arms"][m]["n"] == 12 for m in models)
    out["complete"] = complete
    cap = out["contrasts"]["MLP capacity: dnn_large - dnn"]["different"]
    out["stage2_mlp"] = "dnn_large" if cap else "dnn"
    (EXP / "analysis_s1.json").write_text(json.dumps(out, indent=1, default=str))
    c = out["contrasts"]
    fmt = lambda k: f"{k} median {c[k]['median_diff']:+.1f} ({c[k]['sign']['positive']}/{c[k]['sign']['n']} pairs positive)"
    detail = ("; ".join(fmt(k) for k in c) + f". Capacity {'matters' if cap else 'does not matter'} for the MLP, "
              f"so Stage 2's MLP is {out['stage2_mlp']}.")
    if complete:
        set_gate("G1", "pass", detail)
    print("complete" if complete else "partial", detail)


def w0() -> None:
    bdir = RES / "results_p3s0_battery"
    rows = [json.loads(p.read_text()) for p in sorted(bdir.glob("*.json"))]
    cells: dict[str, list[dict]] = {}
    for r in rows:
        cond = "none" if r["delay"] == "none" else f"fixed{int(r['delay_value'] or 0)}_aug{int(bool(r['augment_obs']))}"
        cells.setdefault(f"{r['env_id']}|{r['model']}|{cond}", []).append(r)
    metrics = ["effort", "abs_du", "abs_d2u", "saturation", "reversals_per_s", "hf_frac", "work", "rms_theta_2nd_half"]
    out = {"n_checkpoints": len(rows),
           "param_mismatches": [r["source"] for r in rows if r["param_count_rebuilt"] != r["param_count_recorded"]],
           "cells": {}}
    for key, rs in sorted(cells.items()):
        nom_runs = [r["probes"]["nominal"]["episodes"]["return"] for r in rs]
        cell = {"n_seeds": len(rs),
                "nominal_iqm_pooled": round(iqm([x for run in nom_runs for x in run]), 1),
                "nominal_iqm_ci95": boot_iqm(nom_runs),
                "train_final_median": round(statistics.median(r["train_final_avg100_return"] for r in rs), 1),
                "command": {m: round(statistics.median(r["probes"]["nominal"]["mean"][m] for r in rs), 4)
                            for m in metrics if m in rs[0]["probes"]["nominal"]["mean"]}}
        for probe in ["delay_mismatch", "obs_noise", "plant_mass", "plant_length", "dropout"]:
            if probe in rs[0]["probes"]:
                cell[probe] = {k: round(statistics.median(r["probes"][probe][k]["iqm_return"] for r in rs), 1)
                               for k in rs[0]["probes"][probe]}
        out["cells"][key] = cell

    lat_path = RES / "results_p3s0_latency" / "latency.json"
    lat = json.loads(lat_path.read_text()) if lat_path.exists() else None
    if lat:
        out["latency"] = {"host": lat["hostname"], "cpu": lat["cpu"], "gpu": lat["gpu"], "table": {
            k: {"b1_p50_ms": round(v["b1"]["p50_ms"], 3), "b1_p99_ms": round(v["b1"]["p99_ms"], 3),
                "b256_p50_ms": round(v["b256"]["p50_ms"], 3), "max_rate_hz": round(v["b1"]["max_rate_hz_at_p99"]),
                "update_ms": round(v["update_b128"]["p50_ms"], 2)} for k, v in lat["results"].items()}}

    # G4 on Pendulum-v1 checkpoints trained at fixed d=1 + aug.
    pend = {k.split("|")[1]: v for k, v in out["cells"].items() if k.startswith("Pendulum-v1|") and k.endswith("fixed1_aug1")}
    if "dnn" in pend and lat:
        mlp = pend["dnn"]
        verdicts = {}
        for m, c in pend.items():
            if m == "dnn":
                continue
            better_ret = c["nominal_iqm_pooled"] >= mlp["nominal_iqm_pooled"]
            better_du = c["command"]["abs_du"] <= mlp["command"]["abs_du"]
            dominated = not better_ret and not better_du
            p99 = lat["results"].get(f"pendulum_fixed1_aug/{m}/cpu1", {}).get("b1", {}).get("p99_ms")
            admits = p99 is not None and p99 < 10.0
            verdicts[m] = {"non_dominated_vs_mlp": not dominated, "cpu1_p99_ms": p99, "admits_100hz": admits,
                           "passes": (not dominated) and admits}
        out["G4"] = verdicts
        passed = [m for m, v in verdicts.items() if v["passes"]]
        complete = len(rows) >= 102
        detail = ("Passing trunks: " + (", ".join(passed) if passed else "none") + ". " +
                  "; ".join(f"{m}: return {pend[m]['nominal_iqm_pooled']:.0f} vs MLP {mlp['nominal_iqm_pooled']:.0f}, "
                            f"|du| {pend[m]['command']['abs_du']:.3f} vs {mlp['command']['abs_du']:.3f}, "
                            f"CPU p99 {v['cpu1_p99_ms']:.2f} ms" for m, v in verdicts.items() if v["cpu1_p99_ms"] is not None))
        if complete:
            set_gate("G4", "pass" if passed else "fail", detail)
        print(detail)
    (EXP / "analysis_w0.json").write_text(json.dumps(out, indent=1))
    print(f"battery: {len(rows)} checkpoints, {len(out['cells'])} cells, param mismatches {len(out['param_mismatches'])}")


def g2() -> None:
    base = RES / "results_p3s2"
    mlp = json.loads((EXP / "analysis_s1.json").read_text())["stage2_mlp"]
    tags = {"C1": "fixed_aug1_pa", "C3": "stochastic_aug1_pa", "C4": "stochastic_aug0_pa"}
    v = {m: {c: {s: (load_scalar(base / f"Pendulum-v1_ddpg_{m}_{t}_seed{s}.json") or {}).get("final")
                 for s in SEEDS} for c, t in tags.items()} for m in [mlp, "trm", "trm_carryz", "lstm_carry"]}
    v = {m: {c: {s: x for s, x in d.items() if x is not None} for c, d in cs.items()} for m, cs in v.items()}
    gap = lambda a, b, c: {s: v[a][c][s] - v[b][c][s] for s in set(v[a][c]) & set(v[b][c])}
    out = {"mlp": mlp, "medians": {m: {c: round(statistics.median(d.values()), 1) if d else None for c, d in cs.items()}
                                   for m, cs in v.items()}}
    g_c1 = gap("trm_carryz", mlp, "C1")
    passes = {}
    for c in ["C3", "C4"]:
        g_c = gap("trm_carryz", mlp, c)
        out[f"interaction_{c}_minus_C1"] = ct = contrast(g_c, g_c1)
        passes[c] = ct["sign"]["n"] == 6 and ct["sign"]["positive"] >= 5
    for c in tags:
        out[f"carryz_minus_resetz_{c}"] = contrast(v["trm_carryz"][c], v["trm"][c])
        out[f"lstm_carry_minus_mlp_{c}"] = contrast(v["lstm_carry"][c], v[mlp][c])
    complete = all(len(d) == 6 for cs in v.values() for d in cs.values())
    out["complete"] = complete
    out["G2"] = ("pass" if any(passes.values()) else "fail") if complete else "pending"
    g3 = [c for c in tags if out[f"carryz_minus_resetz_{c}"]["sign"]["n"] == 6
          and out[f"carryz_minus_resetz_{c}"]["sign"]["positive"] >= 5]
    out["G3_conditions"] = g3
    (EXP / "analysis_s2.json").write_text(json.dumps(out, indent=1))
    ct3, ct4 = out["interaction_C3_minus_C1"], out["interaction_C4_minus_C1"]
    detail = (f"Interaction C3-C1: median {ct3['median_diff']}, {ct3['sign']['positive']}/{ct3['sign']['n']} seeds positive; "
              f"C4-C1: median {ct4['median_diff']}, {ct4['sign']['positive']}/{ct4['sign']['n']} positive.")
    if complete:
        set_gate("G2", out["G2"], detail)
        set_gate("G3", "pass" if g3 else "fail",
                 f"Carry-Z beats Reset-Z in >=5/6 seeds at: {', '.join(g3) if g3 else 'no condition'}.")
    print(out["G2"], detail)


if __name__ == "__main__":
    {"fetch": fetch, "g0": g0, "g1": g1, "w0": w0, "g2": g2}[sys.argv[1]]()
