"""Phase 5 gate evaluation (rules fixed in gates.json before any data; see phase5/plan/plan.md).

  python gates_eval.py fetch      rsync Phase 5 results from Katana into phase5/results/
  python gates_eval.py gt2        Stage T2 -> experiments/analysis_p5_t2.json

Uses the same statistics helpers as experiments/phase4/gates_eval.py. Records the gate's
status and detail in gates.json when its data are complete; never edits rules, frozen
values or a gate's "notes".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase4"))
import gates_eval as g4  # noqa: E402  (load, contrast, med)

PROJ = HERE.parents[1]
RES = PROJ / "phase5" / "results"
RES4 = PROJ / "phase4" / "results"
EXP = PROJ / "experiments"
GATES = HERE / "gates.json"
KATANA = "katana:/srv/scratch/z5308482/trm-control"
CONDS = {"C0": "none", "C1": "fixed_aug1"}


def gates() -> dict:
    return json.loads(GATES.read_text())


def set_gate(gid: str, status: str, detail: str) -> None:
    g = gates()
    for x in g["gates"]:
        if x["id"] == gid:
            x["status"], x["detail"] = status, detail
    GATES.write_text(json.dumps(g, indent=1) + "\n")


def fetch() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{KATANA}/results_p5t2/", str(RES / "results_p5t2") + "/"], check=False)
    print("results_p5t2:", len(list((RES / "results_p5t2").glob("*.json"))), "files")


def runs(model: str, lr: str) -> dict:
    """{(cond, seed): run} for one arm. TRM at actor LR 1e-3 may come from Stage T (frozen value)."""
    fv = gates()["frozen_values"]
    out = {}
    for c, tag in CONDS.items():
        for s in range(1, 7):
            if model == "trm" and lr == "1e-3" and fv.get("t2_reuse_stage_t_lr1e-3"):
                p = RES4 / "results_p4t" / f"Pendulum-v1_td3_trm_{tag}_seed{s}.json"
            else:
                p = RES / "results_p5t2" / f"Pendulum-v1_td3_{model}_{tag}_alr{lr}_seed{s}.json"
            r = g4.load(p)
            if r:
                out[(c, s)] = r
    return out


def gt2() -> None:
    thr = gates()["frozen_values"]["learned_threshold"]
    arms = {f"trm|{lr}": runs("trm", lr) for lr in ["1e-4", "3e-4", "1e-3"]}
    arms["dnn_large|1e-4"] = runs("dnn_large", "1e-4")
    stage_t_mlp = {}
    for c, tag in CONDS.items():
        for s in range(1, 7):
            r = g4.load(RES4 / "results_p4t" / f"Pendulum-v1_td3_dnn_large_{tag}_seed{s}.json")
            if r:
                stage_t_mlp[(c, s)] = r
    arms["dnn_large|1e-3 (Stage T)"] = stage_t_mlp
    out = {"threshold": thr, "arms": {}, "contrasts": {}}
    for k, rs in arms.items():
        out["arms"][k] = {
            "n": len(rs), "learned": sum(r["final"] > thr for r in rs.values()),
            "median_final": g4.med(r["final"] for r in rs.values()),
            "finals": {f"{c}|s{s}": round(r["final"], 1) for (c, s), r in sorted(rs.items())},
            "critic_late_median": g4.med(r["critic_late"] for r in rs.values()),
            "eval_last5_median": g4.med(r["eval_last5"] for r in rs.values()),
            "wall_min_median": g4.med(r["wall_s"] / 60 for r in rs.values()),
            "devices": sorted({r["device"] for r in rs.values() if r["device"]})}
    passing = [lr for lr in ["1e-4", "3e-4"] if out["arms"][f"trm|{lr}"]["learned"] >= 10]
    mlp = {k: r["final"] for k, r in arms["dnn_large|1e-4"].items()}
    for lr in passing:
        c = g4.contrast({k: r["final"] for k, r in arms[f"trm|{lr}"].items()}, mlp)
        c["label"] = "LR-matched (primary)" if lr == "1e-4" else "unequal actor LRs (TRM 3e-4 vs MLP 1e-4)"
        out["contrasts"][f"trm@{lr} - dnn_large@1e-4"] = c
    complete = all(v["n"] == 12 for k, v in out["arms"].items())
    out["complete"], out["passing_lrs"] = complete, passing
    out["GT2"] = ("LR explains" if passing else "LR does not explain") if complete else "pending"
    (EXP / "analysis_p5_t2.json").write_text(json.dumps(out, indent=1, default=str))
    learned = "; ".join(f"{k}: {v['learned']}/{v['n']} learned (median {v['median_final']})" for k, v in out["arms"].items())
    con = "; ".join(f"{k}: median {c['median_diff']}, {c['sign']['positive']}/{c['sign']['n']} positive, two-sided p={c['sign']['p_two_sided']:.4f} [{c['label']}]"
                    for k, c in out["contrasts"].items())
    detail = learned + (". " + con if con else "") + "."
    if complete:
        set_gate("GT2", out["GT2"], detail)
    print(out["GT2"], detail)


if __name__ == "__main__":
    {"fetch": fetch, "gt2": gt2}[sys.argv[1]]()
