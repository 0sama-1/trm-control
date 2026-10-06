"""untrained_E for the H1' span (plan §7): mean return of a uniform-random-action
policy, 100 episodes, under the C1 wrapper (fixed d=1 + pending-action queue).

  python untrained_span.py [--episodes 100] [--out untrained_span.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

import gymnasium as gym  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from delays import DelayedEnv, build_delay  # noqa: E402

ENVS = {"Pendulum-v1": 200, "InvertedDoublePendulum-v5": 1000, "Hopper-v5": 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("untrained_span.json"))
    a = ap.parse_args()
    out = {}
    for env_id, max_steps in ENVS.items():
        env = DelayedEnv(gym.make(env_id), build_delay(OmegaConf.create({"name": "fixed", "value": 1})),
                         augment_obs=True)
        env.action_space.seed(0)
        rets = []
        for ep in range(a.episodes):
            env.reset(seed=10_000 + ep)
            total = 0.0
            for _ in range(max_steps):
                _, r, term, trunc, _ = env.step(env.action_space.sample())
                total += float(r)
                if term or trunc:
                    break
            rets.append(total)
        out[env_id] = {"untrained_mean": round(statistics.mean(rets), 2), "sd": round(statistics.stdev(rets), 2),
                       "episodes": a.episodes, "policy": "uniform random actions", "condition": "fixed d=1 + queue"}
        print(env_id, out[env_id])
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
