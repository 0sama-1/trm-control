"""Test-time perturbation wrappers for the Phase 3 evaluation battery
(``projects/trm-delay-robust-scaling/phase3/plan/plan.md`` §4.1, S0.2).

All three sit *below* ``delays.DelayedEnv`` -- directly on the plant -- so they
perturb what the plant does and what its sensors report, never the
pending-action queue the agent itself committed:

    gym.make -> PlantScale -> ActuatorDropout -> ObsNoise -> DelayedEnv

None of them is used in training.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def obs_scale(space: gym.spaces.Box) -> np.ndarray:
    """Per-dimension half-range of the observation space, 1.0 where unbounded.
    Sensor-noise sigma is expressed as a fraction of this."""
    low, high = space.low.astype(np.float64), space.high.astype(np.float64)
    scale = (high - low) / 2.0
    scale[~np.isfinite(scale)] = 1.0
    return scale.astype(np.float32)


class ObsNoise(gym.ObservationWrapper):
    """Additive Gaussian sensor noise, sigma = ``frac`` x the observation's half-range."""

    def __init__(self, env: gym.Env, frac: float, seed: int):
        super().__init__(env)
        self._sigma = frac * obs_scale(env.observation_space)
        self._rng = np.random.RandomState(seed)

    def observation(self, obs):
        noise = self._rng.randn(*np.shape(obs)).astype(np.float32) * self._sigma
        return (np.asarray(obs, dtype=np.float32) + noise).astype(np.float32)


class ActuatorDropout(gym.ActionWrapper):
    """With probability ``p`` per step the command never reaches the actuator,
    which then applies zero effort for that step."""

    def __init__(self, env: gym.Env, p: float, seed: int):
        super().__init__(env)
        self._p = p
        self._rng = np.random.RandomState(seed)

    def action(self, act):
        if self._p > 0 and self._rng.rand() < self._p:
            return np.zeros_like(np.asarray(act, dtype=np.float32))
        return act


class PlantScale(gym.Wrapper):
    """Scales physical plant parameters away from the values the policy trained on.

    Only Pendulum-v1 exposes them as plain attributes (``m``, ``l``) that its
    ``step()`` reads every call, so the scaling takes effect immediately and
    survives ``reset()``. Other environments raise rather than silently
    running unperturbed.
    """

    def __init__(self, env: gym.Env, mass: float = 1.0, length: float = 1.0):
        super().__init__(env)
        base = env.unwrapped
        if not (hasattr(base, "m") and hasattr(base, "l")):
            raise ValueError(f"PlantScale supports Pendulum-v1 only, got {env.spec.id if env.spec else base}")
        base.m = float(base.m) * mass
        base.l = float(base.l) * length


def supports_plant_scale(env_id: str) -> bool:
    return env_id.startswith("Pendulum")
