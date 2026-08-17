from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym
from omegaconf import DictConfig


class Delay(ABC):
    @abstractmethod
    def sample(self, total_steps: int) -> int:
        """Return the number of environment steps to delay the action."""
        ...

    @property
    @abstractmethod
    def max_delay(self) -> int:
        """Upper bound on the delay, used to size the action history buffer."""
        ...


class NoDelay(Delay):
    def sample(self, total_steps: int) -> int:
        return 0

    @property
    def max_delay(self) -> int:
        return 0


class FixedDelay(Delay):
    def __init__(self, value: int):
        self._value = value

    def sample(self, total_steps: int) -> int:
        return self._value

    @property
    def max_delay(self) -> int:
        return self._value


class StochasticDelay(Delay):
    """Resamples delay from a distribution on every step.

    Every distribution's raw sample is clipped to [min_val, max_val] before
    being returned. This isn't optional polish: DelayedEnv sizes its action-
    history buffer as `max_delay + 1` (max_delay == max_val), so an unclipped
    sample above max_val can never be satisfied by the buffer — it would
    silently fall back to a random action (see DelayedEnv) for the rest of
    the episode instead of raising or bounding, which is what `normal` and
    `exponential` used to do before this clip was added.
    """

    def __init__(self, distribution: str, min_val: float, max_val: float, seed: int):
        self._dist = distribution
        self._min = min_val
        self._max = max_val
        self._rng = np.random.RandomState(seed)

    def sample(self, total_steps: int) -> int:
        if self._dist == "uniform":
            raw = self._rng.uniform(self._min, self._max)
        elif self._dist == "normal":
            mean = (self._min + self._max) / 2.0
            std = (self._max - self._min) / 6.0
            raw = self._rng.normal(mean, std)
        elif self._dist == "exponential":
            raw = self._rng.exponential(scale=self._min)
        else:
            raise ValueError(f"Unknown distribution: {self._dist!r}. Choose from: uniform, normal, exponential")
        return int(round(min(max(raw, self._min), self._max)))

    @property
    def max_delay(self) -> int:
        return int(self._max)


class TimeVaryingDelay(Delay):
    """Piecewise-constant delay that steps through a schedule of total training steps."""

    def __init__(self, schedule: list[dict]):
        self._schedule = sorted(schedule, key=lambda e: e["start"])

    def sample(self, total_steps: int) -> int:
        delay = 0
        for entry in self._schedule:
            if total_steps >= entry["start"]:
                delay = int(entry["delay"])
            else:
                break
        return delay

    @property
    def max_delay(self) -> int:
        return max(int(e["delay"]) for e in self._schedule)


class DelayedEnv(gym.Wrapper):
    """Gym wrapper that queues agent actions and applies them with a configurable delay.

    The agent's action is pushed into a history buffer on every step.
    The action applied to the underlying env is the one from `delay.sample()` steps ago.
    If not enough history exists yet (start of episode), a random action is used instead.
    """

    def __init__(self, env: gym.Env, delay: Delay):
        super().__init__(env)
        self._delay = delay
        self._buf: deque = deque(maxlen=delay.max_delay + 1)
        self._global_step = 0

    def reset(self, **kwargs):
        self._buf.clear()
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        d = self._delay.sample(self._global_step)
        self._buf.append(action)

        if d == 0:
            applied = action
        elif len(self._buf) > d:
            applied = list(self._buf)[-(d + 1)]
        else:
            applied = self.env.action_space.sample()

        self._global_step += 1
        return self.env.step(applied)


def build_delay(delay_cfg: DictConfig) -> Delay:
    name = delay_cfg.name
    if name == "none":
        return NoDelay()
    elif name == "fixed":
        return FixedDelay(int(delay_cfg.value))
    elif name == "stochastic":
        return StochasticDelay(
            distribution=delay_cfg.distribution,
            min_val=float(delay_cfg.min),
            max_val=float(delay_cfg.max),
            seed=int(delay_cfg.seed),
        )
    elif name == "time_varying":
        return TimeVaryingDelay([{"start": e.start, "delay": e.delay} for e in delay_cfg.schedule])
    else:
        raise ValueError(f"Unknown delay: {name!r}. Choose from: none, fixed, stochastic, time_varying")
