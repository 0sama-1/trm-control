#!/usr/bin/env python
"""Fixed-seed, exploration-off evaluation of a TRM-control checkpoint.

Produces the mean/std return over N (>=10 recommended) fixed-seed episodes with
exploration off (deterministic / mean actions), plus an optional 95% bootstrap
CI.

Design choices
--------------
* **Reuse, no fork.** The env and the agent are built with the *existing*
  ``Trainer`` infrastructure — ``RLTrainer._make_env`` (so the delay/staleness
  wrappers the policy was trained under are replayed exactly) and the
  trainer's own ``_build_agent`` / ``_load_checkpoint``. No trainer or algo
  file is modified; this script only drives them.
* **Deterministic actions per arch** are derived from each trainer's existing
  deterministic forward path (DDPG/TD3 ``act(..., noise=None)`` / ``noise_std=0``,
  SAC ``Actor.forward`` mean, PPO ``tanh(mu)`` readout) — exploration is off
  by construction.
* **Checkpoint resolution.** ``--ckpt`` points at any ``.pt``; otherwise the
  path is derived from the config identity
  (``models/<env>_<arch>_<model>_<delay>_seed<seed>.pt``). If the checkpoint
  is absent the eval runs on the freshly initialised policy with a warning —
  this is what makes the smoke test runnable without a trained model.
* **wandb.** Disabled by default so ``--help`` and a 1-episode run need no API
  key; pass ``--wandb`` to stream eval metrics + per-run wall-clock/steps-per-sec
  to wandb (plan §5.5 / AGENTS.md §6.5).

Usage
-----
    python pyscripts/evaluate.py --n-episodes 10 arch=ppo model=dnn \
        env=pendulum delay=none seed=42

    # against an explicit checkpoint
    python pyscripts/evaluate.py --n-episodes 50 --ckpt models/...pt \
        arch=sac model=trm env=pendulum delay=stochastic seed=0 --wandb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Maps arch name -> Trainer class. Imported lazily inside main() so that
# ``--help`` works even when torch is unavailable (smoke test guard).
_TRAINER_REGISTRY: dict[str, object] | None = None


def _trainer_registry():
    global _TRAINER_REGISTRY
    if _TRAINER_REGISTRY is None:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from algos.ddpg import DDPGTrainer
        from algos.td3 import TD3Trainer
        from algos.sac import SACTrainer
        from algos.ppo import PPOTrainer
        from utils import set_seed  # noqa: F401  (ensures import side-effects)
        _TRAINER_REGISTRY = {
            "ddpg": DDPGTrainer,
            "td3": TD3Trainer,
            "sac": SACTrainer,
            "ppo": PPOTrainer,
        }
    return _TRAINER_REGISTRY


def _bootstrap_ci(data: np.ndarray, n_boot: int, conf: float, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(data[0]), float(data[0]))
    means = np.mean(rng.choice(data, size=(n_boot, n), replace=True), axis=1)
    alpha = (1.0 - conf) / 2.0
    lo = float(np.percentile(means, 100.0 * alpha))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha)))
    return (lo, hi)


def _build_deterministic_actor(trainer):
    """Return an ``obs -> np.ndarray`` callable with exploration OFF.

    Each branch uses the arch's existing deterministic forward path — no algo
    file is modified.
    """
    import torch  # deferred import: --help must not require torch
    arch = trainer.arch.name
    device = trainer.device

    if arch == "ddpg":
        agent = trainer.agent
        limit = agent.action_limit
        def act(obs):
            return np.clip(agent.act(obs, noise=None), -limit, limit)
        return act

    if arch == "td3":
        agent = trainer.agent
        limit = agent.action_limit
        def act(obs):
            return np.clip(agent.act(obs, noise_std=0.0), -limit, limit)
        return act

    if arch == "sac":
        actor = trainer.agent.actor            # mean readout (deterministic)
        limit = trainer.agent.action_limit

        def act(obs):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = actor(obs_t)
            return np.clip(a.cpu().numpy()[0], -limit, limit)
        return act

    if arch == "ppo":
        actor = trainer.actor
        scale = (trainer.action_high - trainer.action_low) / 2.0
        bias = (trainer.action_high + trainer.action_low) / 2.0

        def act(obs):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, _ = actor(obs_t)
                a = torch.tanh(mu) * scale + bias
            return a.cpu().numpy()[0]
        return act

    raise ValueError(f"No deterministic actor for arch {arch!r}")


def run_eval(trainer, n_episodes: int, seed: int, max_steps_override: int | None = None):
    """Run N deterministic episodes; return (returns, total_steps, wall, steps_per_sec)."""
    import torch  # noqa: F401  (ensure torch loaded; needed by act fn)
    from utils import set_seed

    set_seed(seed)
    act_fn = _build_deterministic_actor(trainer)
    max_steps = int(max_steps_override) if max_steps_override else int(trainer.env_cfg.max_steps)
    cap = max_steps * (int(trainer.cfg.delay.get("max_delay", 1)) + 1) if trainer.cfg.delay.get("max_delay", 1) else max_steps

    returns: list[float] = []
    total_steps = 0
    t0 = time.time()
    for ep in range(n_episodes):
        obs, _ = trainer.env.reset(seed=seed + ep)
        ep_return = 0.0
        steps = 0
        while True:
            action = act_fn(obs)
            obs, reward, terminated, truncated, _ = trainer.env.step(action)
            ep_return += float(reward)
            total_steps += 1
            steps += 1
            if terminated or truncated:
                break
            if steps >= cap:
                break
        returns.append(ep_return)
    wall = time.time() - t0
    steps_per_sec = total_steps / wall if wall > 0 else 0.0
    return returns, total_steps, wall, steps_per_sec


def _identity(cfg) -> dict:
    return {
        "env": cfg.env.id,
        "arch": cfg.arch.name,
        "model": cfg.model.name,
        "delay": cfg.delay.name,
        "seed": cfg.seed,
    }


def _summarize(returns, total_steps, wall, steps_per_sec, identity, ckpt, loaded, ci_samples):
    rng = np.random.default_rng(identity["seed"])
    arr = np.asarray(returns, dtype=float)
    lo, hi = _bootstrap_ci(arr, ci_samples, 0.95, rng)
    n = int(arr.size)
    summary = {
        "condition": identity,
        "checkpoint": str(ckpt),
        "checkpoint_loaded": bool(loaded),
        "n_episodes": n,
        "per_episode_returns": [float(x) for x in arr],
        "mean_return": float(np.mean(arr)) if n else float("nan"),
        "std_return": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "sem": float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "ci95": {"lo": lo, "hi": hi, "n_boot": ci_samples},
        "min_return": float(np.min(arr)) if n else float("nan"),
        "max_return": float(np.max(arr)) if n else float("nan"),
        "eval_wall_clock": float(wall),
        "eval_total_steps": int(total_steps),
        "eval_steps_per_sec": float(steps_per_sec),
        "eval_episodes_per_sec": float(n / wall) if wall > 0 else 0.0,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Fixed-seed, exploration-off evaluation of a TRM-control checkpoint.",
    )
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of eval episodes (>=10 recommended for a stable mean).")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a .pt checkpoint. Default: derive from config identity.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: results/eval_<identity>.json).")
    parser.add_argument("--ci-samples", type=int, default=2000,
                        help="Bootstrap resamples for the 95%% CI.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override per-episode step cap (useful for smoke tests).")
    parser.add_argument("--wandb", action="store_true",
                        help="Stream eval metrics + per-run timing to wandb (needs WANDB_API_KEY).")
    args, overrides = parser.parse_known_args(argv)

    # --- Hydra config composition (same groups as main.py) ---
    try:
        from hydra import initialize_config_dir, compose
    except Exception as exc:  # hydra not installed
        print(f"[evaluate] hydra-core is required: {exc}", file=sys.stderr)
        return 2

    conf_dir = str(REPO_ROOT / "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))

    # Evaluate with wandb managed here (not by the trainer's no-op logger).
    cfg.wandb.enabled = False
    cfg.model_folder = str(REPO_ROOT / cfg.model_folder)
    cfg.video_folder = str(REPO_ROOT / cfg.video_folder)

    if args.n_episodes < 1:
        print("[evaluate] --n-episodes must be >= 1", file=sys.stderr)
        return 2

    identity = _identity(cfg)
    print(f"[evaluate] condition: {identity}")

    # --- Construct the real trainer (env + agent) and load weights ---
    registry = _trainer_registry()
    trainer_cls = registry[cfg.arch.name]
    trainer = trainer_cls(cfg)          # builds env + no-op logger + model_path
    trainer._build_agent()              # reuses the trainer's agent construction

    ckpt_path = args.ckpt or trainer.model_path
    loaded = False
    if Path(ckpt_path).exists():
        trainer.model_path = ckpt_path
        trainer._load_checkpoint()
        loaded = True
        print(f"[evaluate] loaded checkpoint: {ckpt_path}")
    else:
        print(f"[evaluate] WARNING: no checkpoint at {ckpt_path}; "
              f"evaluating the random-init policy (smoke-test mode).")

    # --- Run evaluation ---
    try:
        returns, total_steps, wall, steps_per_sec = run_eval(
            trainer, args.n_episodes, int(cfg.seed), args.max_steps
        )
    finally:
        trainer.env.close()

    summary = _summarize(returns, total_steps, wall, steps_per_sec,
                         identity, ckpt_path, loaded, args.ci_samples)

    # --- Persist summary JSON ---
    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"eval_{identity['env']}_{identity['arch']}_"
        f"{identity['model']}_{identity['delay']}_seed{identity['seed']}_"
        f"n{args.n_episodes}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[evaluate] wrote {out_path}")

    # --- Optional wandb logging (per-run wall-clock + steps/sec, §5.5/§6.5) ---
    if args.wandb:
        try:
            import wandb
            run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity or None,
                name=f"eval_{identity['env']}_{identity['arch']}_{identity['model']}_"
                     f"{identity['delay']}_seed{identity['seed']}",
                group=identity["env"],
                job_type=identity["arch"],
                config={"condition": identity, "checkpoint": str(ckpt_path),
                        "n_episodes": args.n_episodes, "loaded": loaded},
                tags=[identity["arch"], identity["env"], identity["model"]],
            )
            run.log({
                "eval/mean_return": summary["mean_return"],
                "eval/std_return": summary["std_return"],
                "eval/ci95_lo": summary["ci95"]["lo"],
                "eval/ci95_hi": summary["ci95"]["hi"],
                "eval/min_return": summary["min_return"],
                "eval/max_return": summary["max_return"],
                "eval/wall_clock": summary["eval_wall_clock"],
                "eval/steps_per_sec": summary["eval_steps_per_sec"],
                "eval/episodes_per_sec": summary["eval_episodes_per_sec"],
                "eval/n_episodes": args.n_episodes,
            }, step=0)
            # per-episode table for inspection
            run.log({"eval/returns": [wandb.plot.bar(
                [[i, r] for i, r in enumerate(returns)],
            )]}, step=0) if False else None
            run.finish()
            print("[evaluate] logged eval metrics to wandb")
        except Exception as exc:
            print(f"[evaluate] wandb logging failed (continuing): {exc}", file=sys.stderr)

    # --- Console report ---
    m = summary["mean_return"]
    s = summary["std_return"]
    lo, hi = summary["ci95"]["lo"], summary["ci95"]["hi"]
    print("\n================ EVALUATION SUMMARY ================")
    print(f"  condition : {identity['arch']} | {identity['model']} | "
          f"{identity['env']} | delay={identity['delay']} | seed={identity['seed']}")
    print(f"  checkpoint: {'loaded' if loaded else 'NONE (random init)'}")
    print(f"  episodes  : {args.n_episodes}")
    print(f"  mean return : {m:9.3f}")
    print(f"  std  return : {s:9.3f}")
    print(f"  95% CI (bootstrap, {args.ci_samples} resamples): [{lo:9.3f}, {hi:9.3f}]")
    print(f"  steps/sec : {summary['eval_steps_per_sec']:8.2f}  "
          f"(wall {summary['eval_wall_clock']:.2f}s, {total_steps} steps)")
    print("===================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
