"""Phase 3 Stage 0.2 -- checkpoint evaluation battery (plan §4.1, S0.2).

Scores every banked checkpoint, with no retraining, on:
  nominal         20 deterministic episodes on unseen seeds
  delay_mismatch  test-time fixed delay d in {0..5} (observation layout kept as trained)
  obs_noise       Gaussian sensor noise, sigma in {0.01, 0.05, 0.10} x obs half-range
  plant_mass      Pendulum m x {0.7, 0.85, 1.15, 1.3}           (Pendulum only)
  plant_length    Pendulum l x {0.7, 0.85, 1.15, 1.3}           (Pendulum only)
  dropout         actuator command lost with p in {0.05, 0.10}
Every grid point reports return plus the command-quality suite (src/eval.py).
Sigma = 0, d = trained, scale = 1 and p = 0 are all the nominal point.

Usage (from anywhere; the repo root is found by walking up to main.py):
  python w0_eval.py --list battery_list.txt --shard 0 --nshards 17 --out results_p3s0_battery
  python w0_eval.py --results results_v3/Pendulum-v1_ddpg_trm_fixed_seed1.json --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    while not (p / "main.py").exists():
        if p.parent == p:
            raise RuntimeError("could not find the repo root (main.py) above this script")
        p = p.parent
    return p


REPO = _repo_root()
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from eval import load_actor, load_summary, make_env, run_episodes, summarise, Policy  # noqa: E402
from perturb import supports_plant_scale  # noqa: E402

NOMINAL_SEEDS = [2_000_000 + i for i in range(20)]
GRID_SEEDS = [3_000_000 + i for i in range(10)]
GRID = {
    "delay_mismatch": [0, 1, 2, 3, 4, 5],
    "obs_noise": [0.01, 0.05, 0.10],
    "plant_mass": [0.7, 0.85, 1.15, 1.3],
    "plant_length": [0.7, 0.85, 1.15, 1.3],
    "dropout": [0.05, 0.10],
}


def evaluate(summary_path: Path, models_dir: Path, device: str, quick: bool = False) -> dict:
    t0 = time.time()
    summary = load_summary(summary_path)
    actor, cfg, rebuilt = load_actor(summary, models_dir, device)
    policy = Policy(actor, device)
    env_id = cfg.env.id
    max_steps = int(cfg.env.max_steps)
    nominal_seeds = NOMINAL_SEEDS[:4] if quick else NOMINAL_SEEDS
    grid_seeds = GRID_SEEDS[:2] if quick else GRID_SEEDS

    def point(seeds, **kw):
        eps = run_episodes(policy, lambda i: make_env(cfg, noise_seed=seeds[i], **kw), seeds, max_steps, env_id)
        return summarise(eps)

    probes: dict = {"nominal": point(nominal_seeds)}
    probes["delay_mismatch"] = {str(d): point(grid_seeds, test_delay=d) for d in GRID["delay_mismatch"]}
    probes["obs_noise"] = {str(s): point(grid_seeds, obs_noise=s) for s in GRID["obs_noise"]}
    if supports_plant_scale(env_id):
        probes["plant_mass"] = {str(m): point(grid_seeds, mass=m) for m in GRID["plant_mass"]}
        probes["plant_length"] = {str(l): point(grid_seeds, length=l) for l in GRID["plant_length"]}
    probes["dropout"] = {str(p): point(grid_seeds, dropout=p) for p in GRID["dropout"]}

    return {
        "source": summary_path.name,
        "model_id": summary["model_id"],
        "env_id": env_id,
        "arch": summary["arch"],
        "model": summary["model"],
        "delay": summary["delay"],
        "delay_value": summary.get("delay_value"),
        "augment_obs": summary["augment_obs"],
        "run_tag": summary.get("run_tag", ""),
        "seed": summary["seed"],
        "train_final_avg100_return": summary["final_avg100_return"],
        "train_wall_clock_seconds": summary["wall_clock_seconds"],
        "param_count_recorded": summary["param_count"],
        "param_count_rebuilt": rebuilt,
        "device": device,
        "quick": quick,
        "eval_seconds": time.time() - t0,
        "probes": probes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=[], help="results JSON files to evaluate")
    ap.add_argument("--list", help="file with one results JSON path per line (relative to the repo root)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results_p3s0_battery")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true", help="smoke test: 4 nominal + 2 per grid point")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = [Path(p) for p in args.results]
    if args.list:
        paths += [Path(l.strip()) for l in open(args.list) if l.strip()]
    paths = [p.resolve() if (p.is_absolute() or p.exists()) else REPO / p for p in paths]
    paths = paths[args.shard::args.nshards]
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models)
    models_dir = models_dir if models_dir.is_absolute() else REPO / models_dir

    torch.set_num_threads(2)
    failures = 0
    for p in paths:
        dest = out_dir / p.name
        if dest.exists() and not args.force:
            print(f"skip (done) {p.name}", flush=True)
            continue
        try:
            res = evaluate(p, models_dir, args.device, quick=args.quick)
        except Exception as e:  # keep going: one bad checkpoint must not sink the shard
            failures += 1
            print(f"FAIL {p.name}: {type(e).__name__}: {e}", flush=True)
            continue
        ok = "OK" if res["param_count_rebuilt"] == res["param_count_recorded"] else "PARAM-MISMATCH"
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(res, f)
        tmp.rename(dest)
        nom = res["probes"]["nominal"]
        print(f"{ok} {p.name}: nominal IQM {nom['iqm_return']:.1f} | |du| {nom['mean']['abs_du']:.4f} "
              f"| {res['eval_seconds']:.0f}s", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
