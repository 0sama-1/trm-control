"""Compact per-run cache of every Phase 3 training run (and the banked Phase 1 runs
that Stage 1 re-runs), so the analysis never re-parses the ~30 MB per-step histories.

  python extract_runs.py   -> results_phase3/runs_cache.json
"""

from __future__ import annotations

import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "results_phase3"
SOURCES = {
    "s0h": RES / "results_p3s0",
    "s1": RES / "results_p3s1",
    "s2": RES / "results_p3s2",
    "s2x": RES / "results_p3s2x",
    "p1": PROJ / "results_v3",
}


def compact(path: Path, stage: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    hist = d["episode_history"]
    cfg = d.get("config", {})
    return {
        "stage": stage, "file": path.name, "env": d["env_id"], "model": d["model"], "delay": d["delay"],
        "aug": bool(d["augment_obs"]), "tag": d.get("run_tag", ""), "seed": d["seed"],
        "prev_action": bool(cfg.get("prev_action_obs", False)),
        "final": d["final_avg100_return"], "wall_s": d["wall_clock_seconds"], "params": d["param_count"],
        "device": d.get("device_name"), "model_id": d["model_id"],
        "ret": [r["return"] for r in hist],
        "critic": [r.get("critic_loss") for r in hist],
        "actor": [r.get("actor_loss") for r in hist],
        "du": [r.get("mean_action_delta") for r in hist],
        "eval": [[r["episode"], r["eval_return"]] for r in hist if "eval_return" in r],
    }


def main() -> None:
    out = []
    for stage, folder in SOURCES.items():
        files = sorted(folder.glob("*.json")) if folder.exists() else []
        if stage == "p1":
            files = [f for f in files if f.name.startswith("Pendulum")]
        for f in files:
            out.append(compact(f, stage))
        print(stage, len(files))
    (RES / "runs_cache.json").write_text(json.dumps(out))
    print("runs", len(out))


if __name__ == "__main__":
    main()
