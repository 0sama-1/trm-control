"""Observation-staleness wrapper for gymnasium environments.

Bypasses the action-delay mechanism (handled by DelayedEnv in delays.py).
Instead, it buffers the last `k` observations and returns the stale (delayed)
one on each step — a direct test of the H5 "high-level latent smooths stale
input" mechanism.

Composable with DelayedEnv: wrap in either order.
"""

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig


class StaleObservationWrapper(gym.Wrapper):
    """Gym wrapper that buffers the last `k` observations and returns the stale one.

    The agent's observation on step `t` is the observation from step `t - k`.
    If not enough history exists yet (start of episode), the current observation
    is returned instead (no padding).

    Composes independently of DelayedEnv (action delay). The two wrappers
    can be stacked in any order; the observation-staleness only affects what
    the agent *sees*, not what the env *receives*.
    """

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self._k = k
        self._buf: deque = deque(maxlen=k + 1)

    def reset(self, **kwargs):
        self._buf.clear()
        obs, info = self.env.reset(**kwargs)
        self._buf.append(obs)
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buf.append(obs)

        if len(self._buf) > self._k:
            stale_obs = list(self._buf)[-(self._k + 1)]
        else:
            stale_obs = obs  # not enough history yet — return current

        return stale_obs, reward, terminated, truncated, info


def build_staleness(staleness_cfg: DictConfig) -> int:
    """Return the staleness `k` from the config.

    Returns 0 for 'none' (StaleObservationWrapper with k=0 is a no-op:
    the buffer always returns the current observation).
    """
    name = staleness_cfg.name
    if name == "none":
        return 0
    elif name == "fixed":
        return int(staleness_cfg.value)
    elif name == "stochastic":
        raise NotImplementedError("Stochastic observation staleness not yet implemented")
    elif name == "time_varying":
        raise NotImplementedError("Time-varying observation staleness not yet implemented")
    else:
        raise ValueError(
            f"Unknown staleness: {name!r}. Choose from: none, fixed, stochastic, time_varying"
        )