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

    With `augment_obs=True`, the observation returned to the agent is extended with
    the queue of actions already committed but not yet applied to the plant — sized
    to `delay.max_delay` actions, zero-padded when fewer are pending (start of
    episode, or a per-step delay below `max_delay`). Without this, the raw
    observation doesn't determine the transition on its own — see
    `augmentation.md` for why that matters and what this fixes. A no-op when
    `max_delay == 0` (nothing to expose) or `augment_obs=False` (the default,
    preserving the original observation for existing configs/checkpoints).
    """

    def __init__(self, env: gym.Env, delay: Delay, augment_obs: bool = False):
        super().__init__(env)
        self._delay = delay
        self._max_delay = delay.max_delay
        self._buf: deque = deque(maxlen=self._max_delay + 1)
        self._global_step = 0

        self._action_dim = (
            int(np.prod(env.action_space.shape)) if isinstance(env.action_space, gym.spaces.Box) else 1
        )
        self._augment = augment_obs and self._max_delay > 0
        if self._augment:
            self.observation_space = self._build_augmented_space(env)

    def _build_augmented_space(self, env: gym.Env) -> gym.spaces.Box:
        obs_space = env.observation_space
        if isinstance(env.action_space, gym.spaces.Box):
            act_low  = np.tile(env.action_space.low.astype(np.float32),  self._max_delay)
            act_high = np.tile(env.action_space.high.astype(np.float32), self._max_delay)
        else:
            # Discrete: represent each queued action as a single scalar index.
            act_low  = np.zeros(self._max_delay, dtype=np.float32)
            act_high = np.full(self._max_delay, env.action_space.n - 1, dtype=np.float32)
        return gym.spaces.Box(
            low=np.concatenate([obs_space.low.astype(np.float32),  act_low]),
            high=np.concatenate([obs_space.high.astype(np.float32), act_high]),
            dtype=np.float32,
        )

    def _pending_actions(self) -> np.ndarray:
        """Actions already committed but not yet applied, oldest-pending first,
        zero-padded to `max_delay` slots.

        Exact for constant delay (`FixedDelay`/`NoDelay`): `step()`'s `elif`
        branch always applies the buffer's oldest entry once it's full, so
        dropping that entry here leaves exactly what's still queued. For
        `StochasticDelay`/`TimeVaryingDelay`, whose per-step delay can be
        *less* than `max_delay`, the applied action isn't always the oldest
        one in the buffer — this is a best-effort approximation there (the
        `max_delay` most recently committed actions), not a proven sufficient
        statistic. See `augmentation.md` §"Known limitation".
        """
        pending = list(self._buf)
        if len(pending) > self._max_delay:
            pending = pending[1:]
        vecs = [np.asarray(a, dtype=np.float32).reshape(-1) for a in pending]
        pad = self._max_delay - len(vecs)
        if pad > 0:
            vecs = vecs + [np.zeros(self._action_dim, dtype=np.float32) for _ in range(pad)]
        return np.concatenate(vecs) if vecs else np.zeros(0, dtype=np.float32)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._augment:
            return obs
        return np.concatenate([np.asarray(obs, dtype=np.float32), self._pending_actions()]).astype(np.float32)

    def reset(self, **kwargs):
        self._buf.clear()
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

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
        next_obs, reward, terminated, truncated, info = self.env.step(applied)
        # Expose the actually-applied (post-delay) action and the sampled
        # delay -- invisible before this, and directly relevant to this
        # project's central premise (the delay confound, see
        # augmentation.md). Non-breaking: additive info keys only.
        info = dict(info)
        info["delay_applied_action"] = np.asarray(applied, dtype=np.float32).tolist()
        info["delay_sampled_steps"] = d
        return self._augment_obs(next_obs), reward, terminated, truncated, info


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
