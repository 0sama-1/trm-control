"""Stage E (plan §8, exploratory): the actor's local Lipschitz constant -- the spectral
norm of da/ds -- over the states visited in deterministic evaluation episodes.

  python lipschitz.py --list LIST.txt --models DIR --out DIR [--shard i --nshards n] [--episodes 20]

LIST.txt holds one results-JSON path per line (Phase 3 Stage 1 runs). For each
checkpoint the actor is rebuilt (src/eval.py), rolled out deterministically for
`episodes` episodes on the battery's unseen seeds under its training condition, and
at every visited state the Jacobian da/ds (act_dim x obs_dim, autograd) is taken over
the *plant-observation* inputs only (the pending-action queue and any previous-action
slot are held fixed: they are the policy's own past outputs, not sensor readings).
Writes one JSON per checkpoint: spectral-norm median / p95 / max, plus the full-input
spectral norm for reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# The repo root locally, or the frozen snapshot root on Katana (p4code/<stage>/): whichever ancestor holds src/eval.py.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "src" / "eval.py").exists())
sys.path.insert(0, str(REPO / "src"))

from eval import load_actor, load_summary, make_env  # noqa: E402

# The battery's nominal unseen seeds (experiments/phase3/w0_eval.py NOMINAL_SEEDS), so the
# visited states match the episodes behind the Phase 3 reversal and noise-drop numbers.
NOMINAL_SEEDS = [2_000_000 + i for i in range(20)]


def jacobian_norms(actor, states: np.ndarray, plant_dim: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    # TRM's truncated forward runs all but the last outer cycle under no_grad, which would
    # hide those cycles from the Jacobian. full_grad computes the same function without
    # the truncation, so the Jacobian is that of the deployed input-output map.
    if hasattr(actor.trunk, "full_grad"):
        actor.trunk.full_grad = True
    x = torch.tensor(states, dtype=torch.float32, device=device)
    full, plant = [], []
    for i in range(0, len(x), 256):
        xb = x[i:i + 256]
        J = torch.func.vmap(torch.func.jacrev(lambda s: actor(s.unsqueeze(0))[0]))(xb)   # (B, act, obs)
        full.append(torch.linalg.matrix_norm(J, ord=2).detach().cpu().numpy())
        plant.append(torch.linalg.matrix_norm(J[:, :, :plant_dim], ord=2).detach().cpu().numpy())
    return np.concatenate(full), np.concatenate(plant)


def run_one(path: Path, models: Path, episodes: int, device: str) -> dict:
    summary = load_summary(path)
    actor, cfg, _ = load_actor(summary, models, device)
    if getattr(actor, "stateful", False):
        raise ValueError("stateful trunks are out of Stage E's scope")
    env = make_env(cfg)
    import gymnasium as gym
    plant_dim = gym.make(cfg.env.id).observation_space.shape[0]
    states = []
    for seed in NOMINAL_SEEDS[:episodes]:
        obs, _ = env.reset(seed=seed)
        for _ in range(cfg.env.max_steps):
            states.append(np.asarray(obs, dtype=np.float32))
            with torch.no_grad():
                a = actor(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0]
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    env.close()
    full, plant = jacobian_norms(actor, np.stack(states), plant_dim, device)
    q = lambda v, p: float(np.percentile(v, p))
    return {"source": path.name, "model_id": summary["model_id"], "model": summary["model"], "seed": summary["seed"],
            "delay": summary["delay"], "n_states": int(len(states)), "plant_obs_dim": plant_dim,
            "lip_plant": {"median": q(plant, 50), "p95": q(plant, 95), "max": float(plant.max())},
            "lip_full": {"median": q(full, 50), "p95": q(full, 95), "max": float(full.max())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = [Path(l.strip()) for l in a.list.read_text().splitlines() if l.strip()][a.shard::a.nshards]
    for p in paths:
        dest = a.out / p.name
        if dest.exists():
            continue
        row = run_one(p, a.models, a.episodes, device)
        dest.write_text(json.dumps(row, indent=1) + "\n")
        print(p.name, round(row["lip_plant"]["median"], 3), flush=True)


if __name__ == "__main__":
    main()
