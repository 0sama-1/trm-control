"""Checkpoint evaluation harness for Phase 3 (plan §4.1 S0.2, engineering item E-1).

Rebuilds a trained policy from its results JSON + checkpoint, then runs
deterministic (noise-free) rollouts under a grid of test-time conditions and
scores each episode on return *and* on control quality -- the deployment
axis the project had never measured (plan §3).

Episodes within a grid point run as one batch of parallel environments, so the
actor sees a (n_envs, obs_dim) batch per step; every checkpoint is scored on
the same episode seeds (common random numbers), so arm-vs-arm comparisons at a
grid point are paired.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from delays import DelayedEnv, build_delay
from perturb import ActuatorDropout, ObsNoise, PlantScale

REPO_ROOT = Path(__file__).resolve().parent.parent
CONF = REPO_ROOT / "conf"

# Control period per env step (s). MountainCarContinuous has no physical time
# base, so its "per second" rates are per step.
CONTROL_DT = {"Pendulum-v1": 0.05, "MountainCarContinuous-v0": 1.0, "BipedalWalker-v3": 0.02,
              "InvertedDoublePendulum-v5": 0.05}


# ---------------------------------------------------------------------------
# Rebuilding a policy from a results file
# ---------------------------------------------------------------------------

def _env_cfg_for(env_id: str) -> DictConfig:
    for f in sorted((CONF / "env").glob("*.yaml")):
        c = OmegaConf.load(f)
        if c.id == env_id:
            return c
    raise ValueError(f"no conf/env/*.yaml with id {env_id!r}")


def run_config(summary: dict) -> DictConfig:
    """The training config of a run. Phase 3 results embed it; Phase 1-2
    results predate that, so it is rebuilt from the identity fields they do
    carry plus the config files those runs used unmodified (the rebuilt
    parameter count is checked against the recorded one by the caller)."""
    if "config" in summary:
        return OmegaConf.create(summary["config"])
    delay = {"name": summary["delay"]}
    if summary["delay"] == "fixed":
        delay["value"] = summary["delay_value"]
    elif summary["delay"] != "none":
        raise ValueError(f"cannot rebuild delay config {summary['delay']!r} without an embedded config")
    return OmegaConf.create({
        "arch": OmegaConf.to_container(OmegaConf.load(CONF / "arch" / f"{summary['arch']}.yaml")),
        "model": OmegaConf.to_container(OmegaConf.load(CONF / "model" / f"{summary['model']}.yaml")),
        "env": OmegaConf.to_container(_env_cfg_for(summary["env_id"])),
        "delay": delay,
        "augment_obs": summary["augment_obs"],
        "seed": summary["seed"],
    })


def _actor_critic_classes(arch: str):
    if arch == "ddpg":
        from algos.ddpg import Actor, Critic
    elif arch == "td3":
        from algos.td3 import Actor, Critic
    else:
        raise ValueError(f"evaluation supports deterministic-actor algorithms (ddpg, td3), got {arch!r}")
    return Actor, Critic


def load_actor(summary: dict, models_dir: Path, device: str):
    """Returns (actor, cfg, rebuilt_param_count). The actor is in eval mode on `device`."""
    cfg = run_config(summary)
    env = make_env(cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    limit = float(env.action_space.high[0])
    env.close()

    Actor, Critic = _actor_critic_classes(cfg.arch.name)
    actor = Actor(obs_dim, act_dim, limit, cfg.model)
    n_critics = 2 if cfg.arch.name == "td3" else 1
    critic_params = sum(p.numel() for p in Critic(obs_dim, act_dim, cfg.model).parameters())
    rebuilt = sum(p.numel() for p in actor.parameters()) + n_critics * critic_params

    ckpt = torch.load(models_dir / f"{summary['model_id']}.pt", map_location="cpu")
    actor.load_state_dict(ckpt["actor"])
    actor.to(device).eval()
    return actor, cfg, rebuilt


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

def trained_slots(cfg: DictConfig) -> int:
    """Pending-action slots the policy was trained to read (0 if unaugmented)."""
    if not cfg.augment_obs:
        return 0
    return build_delay(cfg.delay).max_delay


def make_env(
    cfg: DictConfig,
    test_delay: int | None = None,
    obs_noise: float = 0.0,
    dropout: float = 0.0,
    mass: float = 1.0,
    length: float = 1.0,
    noise_seed: int = 0,
) -> gym.Env:
    """The run's training environment, optionally perturbed. `test_delay`
    replaces the training delay with a fixed test-time delay while keeping the
    observation layout (number of pending-action slots) the policy trained on."""
    env = gym.make(cfg.env.id)
    wrap = _prev_action if cfg.get("prev_action_obs", False) else (lambda e: e)
    if mass != 1.0 or length != 1.0:
        env = PlantScale(env, mass=mass, length=length)
    if dropout > 0:
        env = ActuatorDropout(env, dropout, seed=noise_seed + 7)
    if obs_noise > 0:
        env = ObsNoise(env, obs_noise, seed=noise_seed + 13)
    expose = bool(cfg.delay.get("expose_realised", False))
    if test_delay is None:
        delay = build_delay(cfg.delay)
        return wrap(DelayedEnv(env, delay, augment_obs=bool(cfg.augment_obs), expose_realised=expose))
    from delays import FixedDelay, NoDelay
    delay = NoDelay() if test_delay == 0 else FixedDelay(int(test_delay))
    slots = trained_slots(cfg)
    return wrap(DelayedEnv(env, delay, augment_obs=slots > 0, expose_realised=expose, aug_slots=slots))


def _prev_action(env: gym.Env) -> gym.Env:
    from delays import PrevActionObs
    return PrevActionObs(env)


def _plant_signals(env_id: str, base) -> tuple[np.ndarray, float | None]:
    """(actuated-joint angular velocity per action dim, pendulum angle or None),
    read from the true plant state rather than the (possibly noisy) observation."""
    if env_id.startswith("Pendulum"):
        th, thdot = base.state
        return np.array([thdot], dtype=np.float64), float(((th + np.pi) % (2 * np.pi)) - np.pi)
    if env_id.startswith("MountainCar"):
        return np.array([base.state[1]], dtype=np.float64), None
    if env_id.startswith("BipedalWalker"):
        return np.array([j.speed for j in base.joints], dtype=np.float64), None
    if env_id.startswith("InvertedDoublePendulum"):
        # Actuator drives the cart: power = force x cart velocity. Angle = upper pole from vertical.
        qpos, qvel = base.data.qpos, base.data.qvel
        return np.array([qvel[0]], dtype=np.float64), float(qpos[1] + qpos[2])
    return np.zeros(1), None


# ---------------------------------------------------------------------------
# Policy + rollouts
# ---------------------------------------------------------------------------

class Policy:
    """Deterministic batched policy. Trunks that carry a latent across steps
    expose `initial_carry(batch)` and return `(action, carry)` from forward;
    stateless trunks ignore reset()."""

    def __init__(self, actor, device: str):
        self.actor = actor
        self.device = device
        self.stateful = bool(getattr(actor, "stateful", False))
        self.carry = None

    def reset(self, n: int) -> None:
        if self.stateful:
            self.carry = self.actor.initial_carry(n, self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self.stateful:
            a, self.carry = self.actor(x, self.carry)
        else:
            a = self.actor(x)
        return a.cpu().numpy()


def run_episodes(policy: Policy, env_factory, seeds: list[int], max_steps: int, env_id: str) -> list[dict]:
    envs = [env_factory(i) for i in range(len(seeds))]
    limit = float(envs[0].action_space.high[0])
    for e, s in zip(envs, seeds):
        # DelayedEnv fills the first d steps with action_space.sample(); seed it
        # so every grid point is reproducible run to run.
        e.action_space.seed(s)
    first = [e.reset(seed=s)[0] for e, s in zip(envs, seeds)]
    obs = np.stack(first).astype(np.float32)
    policy.reset(len(envs))
    active = np.ones(len(envs), dtype=bool)
    logs = [{"cmd": [], "applied": [], "reward": [], "omega": [], "theta": []} for _ in envs]
    for _ in range(max_steps):
        acts = policy.act(obs)
        for i in np.flatnonzero(active):
            omega, theta = _plant_signals(env_id, envs[i].unwrapped)
            o, r, term, trunc, info = envs[i].step(acts[i])
            lg = logs[i]
            lg["cmd"].append(acts[i].astype(np.float64))
            lg["applied"].append(np.asarray(info["delay_applied_action"], dtype=np.float64))
            lg["reward"].append(float(r))
            lg["omega"].append(omega)
            if theta is not None:
                lg["theta"].append(theta)
            obs[i] = o
            if term or trunc:
                active[i] = False
        if not active.any():
            break
    for e in envs:
        e.close()
    dt = CONTROL_DT.get(env_id, 1.0)
    return [episode_metrics(lg, dt, limit) for lg in logs]


def _sign_reversals(u: np.ndarray, deadband: float) -> int:
    """Direction reversals of one command channel, holding the last sign
    through a small deadband so jitter around zero is not counted."""
    count, last = 0, 0.0
    for x in u:
        if abs(x) <= deadband:
            continue
        s = 1.0 if x > 0 else -1.0
        if last != 0.0 and s != last:
            count += 1
        last = s
    return count


def _hf_fraction(u: np.ndarray, dt: float) -> float:
    """Fraction of (mean-removed) command power above fs/8 = Nyquist/4."""
    if len(u) < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(u - u.mean())) ** 2
    freqs = np.fft.rfftfreq(len(u), dt)
    total = spec[1:].sum()
    if total <= 0:
        return 0.0
    return float(spec[freqs > (0.5 / dt) / 4].sum() / total)


def episode_metrics(lg: dict, dt: float, limit: float) -> dict:
    cmd = np.asarray(lg["cmd"])            # (T, A) controller command
    applied = np.asarray(lg["applied"])    # (T, A) what the actuator received
    omega = np.asarray(lg["omega"])        # (T, A') joint angular velocity
    T = len(cmd)
    du = np.diff(cmd, axis=0)
    d2u = np.diff(cmd, n=2, axis=0)
    out = {
        "return": float(np.sum(lg["reward"])),
        "length": T,
        "effort": float(np.mean(cmd ** 2)),                                   # (1/T) sum u^2
        "abs_du": float(np.mean(np.abs(du))) if len(du) else 0.0,              # command rate
        "abs_d2u": float(np.mean(np.abs(d2u))) if len(d2u) else 0.0,           # command jerk
        "saturation": float(np.mean(np.abs(cmd) >= 0.99 * limit)),
        "reversals_per_s": float(sum(_sign_reversals(cmd[:, k], 0.01 * limit) for k in range(cmd.shape[1]))
                                 / cmd.shape[1] / (T * dt)),
        "hf_frac": float(np.mean([_hf_fraction(cmd[:, k], dt) for k in range(cmd.shape[1])])),
    }
    if omega.shape[1] == applied.shape[1]:
        out["work"] = float(np.sum(np.abs(applied * omega)) * dt)             # sum |u . omega| dt
    if lg["theta"]:
        th = np.asarray(lg["theta"])
        out["rms_theta_2nd_half"] = float(np.sqrt(np.mean(th[T // 2:] ** 2)))
    return out


def summarise(episodes: list[dict]) -> dict:
    keys = episodes[0].keys()
    returns = np.sort([e["return"] for e in episodes])
    n = len(returns)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return {
        "n": n,
        "iqm_return": float(np.mean(returns[lo:hi])) if hi > lo else float(np.mean(returns)),
        "mean": {k: float(np.mean([e[k] for e in episodes])) for k in keys},
        "episodes": {k: [e[k] for e in episodes] for k in keys},
    }


def iqm(x) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(np.mean(x[lo:hi])) if hi > lo else float(np.mean(x))


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
