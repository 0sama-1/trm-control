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

    def on_reset(self) -> None:
        """Called by DelayedEnv at every episode reset (no-op for most delays)."""

    def peek(self) -> int:
        """The delay an oracle may reveal before the first step of an episode
        (0 unless the delay is known in advance, as for EpisodeDelay)."""
        return 0

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
        if self._dist == "uniform_int":
            # Exact discrete uniform over {min, ..., max}. The continuous
            # "uniform" branch below rounds a U(min, max) draw, which gives the
            # two endpoints half the mass of the interior values.
            return int(self._rng.randint(int(self._min), int(self._max) + 1))
        if self._dist == "uniform":
            raw = self._rng.uniform(self._min, self._max)
        elif self._dist == "normal":
            mean = (self._min + self._max) / 2.0
            std = (self._max - self._min) / 6.0
            raw = self._rng.normal(mean, std)
        elif self._dist == "exponential":
            raw = self._rng.exponential(scale=self._min)
        else:
            raise ValueError(f"Unknown distribution: {self._dist!r}. Choose from: uniform_int, uniform, normal, exponential")
        return int(round(min(max(raw, self._min), self._max)))

    @property
    def max_delay(self) -> int:
        return int(self._max)


class EpisodeDelay(Delay):
    """Latent per-episode delay (Phase 3 supplementary test): d ~ U{min..max}
    is drawn once at each reset and held for the whole episode. A memoryless
    policy cannot know d; a policy that carries state across steps can
    identify it from a few transitions (system identification) -- the
    condition in which inference has something to recover that no single
    observation reveals."""

    def __init__(self, min_val: int, max_val: int, seed: int):
        self._min, self._max = int(min_val), int(max_val)
        self._rng = np.random.RandomState(seed)
        self._d = self._min

    def on_reset(self) -> None:
        self._d = int(self._rng.randint(self._min, self._max + 1))

    def peek(self) -> int:
        return self._d

    def sample(self, total_steps: int) -> int:
        return self._d

    @property
    def max_delay(self) -> int:
        return self._max


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

    `aug_slots` overrides how many pending-action slots are exposed (default:
    `max_delay`). Only the Phase 3 evaluation battery sets it, to run a policy
    trained at one delay against a different test-time delay while keeping the
    observation layout it was trained on (see `experiments/w0_eval.py`). The
    slots always hold the `aug_slots` most recently committed actions, which is
    identical to the default behaviour when `aug_slots == max_delay`.

    `expose_realised=True` appends the delay realised on the previous step,
    normalised to [0, 1] by `max_delay` (0 at reset), after any augmentation:
    the Phase 3 "delay oracle" arm (plan §4.1, S0.1 arm A3).
    """

    def __init__(
        self,
        env: gym.Env,
        delay: Delay,
        augment_obs: bool = False,
        expose_realised: bool = False,
        aug_slots: int | None = None,
    ):
        super().__init__(env)
        self._delay = delay
        self._max_delay = delay.max_delay
        self._slots = self._max_delay if aug_slots is None else int(aug_slots)
        self._buf: deque = deque(maxlen=max(self._max_delay, self._slots) + 1)
        self._global_step = 0
        self._last_d = 0

        self._action_dim = (
            int(np.prod(env.action_space.shape)) if isinstance(env.action_space, gym.spaces.Box) else 1
        )
        self._augment = augment_obs and self._slots > 0
        self._expose = expose_realised
        if self._augment or self._expose:
            self.observation_space = self._build_augmented_space(env)

    def _build_augmented_space(self, env: gym.Env) -> gym.spaces.Box:
        obs_space = env.observation_space
        lows  = [obs_space.low.astype(np.float32)]
        highs = [obs_space.high.astype(np.float32)]
        if self._augment:
            if isinstance(env.action_space, gym.spaces.Box):
                lows.append(np.tile(env.action_space.low.astype(np.float32),  self._slots))
                highs.append(np.tile(env.action_space.high.astype(np.float32), self._slots))
            else:
                # Discrete: represent each queued action as a single scalar index.
                lows.append(np.zeros(self._slots, dtype=np.float32))
                highs.append(np.full(self._slots, env.action_space.n - 1, dtype=np.float32))
        if self._expose:
            lows.append(np.zeros(1, dtype=np.float32))
            highs.append(np.ones(1, dtype=np.float32))
        return gym.spaces.Box(low=np.concatenate(lows), high=np.concatenate(highs), dtype=np.float32)

    def _pending_actions(self) -> np.ndarray:
        """Actions already committed but not yet applied, oldest-pending first,
        zero-padded to `aug_slots` slots.

        Exact for constant delay (`FixedDelay`/`NoDelay`): `step()`'s `elif`
        branch always applies the buffer's oldest entry once it's full, so
        dropping that entry here leaves exactly what's still queued. For
        `StochasticDelay`/`TimeVaryingDelay`, whose per-step delay can be
        *less* than `max_delay`, the applied action isn't always the oldest
        one in the buffer — this is a best-effort approximation there (the
        `max_delay` most recently committed actions), not a proven sufficient
        statistic. See `augmentation.md` §"Known limitation".
        """
        if self._slots == 0:
            return np.zeros(0, dtype=np.float32)
        pending = list(self._buf)[-self._slots:]
        vecs = [np.asarray(a, dtype=np.float32).reshape(-1) for a in pending]
        pad = self._slots - len(vecs)
        if pad > 0:
            vecs = vecs + [np.zeros(self._action_dim, dtype=np.float32) for _ in range(pad)]
        return np.concatenate(vecs) if vecs else np.zeros(0, dtype=np.float32)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        if not (self._augment or self._expose):
            return obs
        parts = [np.asarray(obs, dtype=np.float32)]
        if self._augment:
            parts.append(self._pending_actions())
        if self._expose:
            parts.append(np.array([self._last_d / max(self._max_delay, 1)], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, **kwargs):
        self._buf.clear()
        self._delay.on_reset()
        self._last_d = self._delay.peek()
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
        self._last_d = d
        next_obs, reward, terminated, truncated, info = self.env.step(applied)
        # Expose the actually-applied (post-delay) action and the sampled
        # delay -- invisible before this, and directly relevant to this
        # project's central premise (the delay confound, see
        # augmentation.md). Non-breaking: additive info keys only.
        info = dict(info)
        info["delay_applied_action"] = np.asarray(applied, dtype=np.float32).tolist()
        info["delay_sampled_steps"] = d
        return self._augment_obs(next_obs), reward, terminated, truncated, info


class PrevActionObs(gym.Wrapper):
    """Appends the agent's previous action (the one it committed last step,
    before any delay; zeros after reset) to the observation.

    Phase 3 Stage 2 input design (plan §10, A-6): every Stage 2 trunk gets
    this input, so memory-carrying and memoryless trunks see identical
    observations and differ only in architecture. Sits outside
    ``DelayedEnv``: it reports what the agent sent, not what the plant got.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        box = env.action_space
        self._zero = np.zeros(int(np.prod(box.shape)), dtype=np.float32)
        self._prev = self._zero
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low.astype(np.float32), box.low.astype(np.float32).reshape(-1)]),
            high=np.concatenate([env.observation_space.high.astype(np.float32), box.high.astype(np.float32).reshape(-1)]),
            dtype=np.float32,
        )

    def _obs(self, obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32), self._prev]).astype(np.float32)

    def reset(self, **kwargs):
        self._prev = self._zero
        obs, info = self.env.reset(**kwargs)
        return self._obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._prev = np.asarray(action, dtype=np.float32).reshape(-1)
        return self._obs(obs), reward, terminated, truncated, info


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
    elif name == "episode":
        return EpisodeDelay(min_val=int(delay_cfg.min), max_val=int(delay_cfg.max), seed=int(delay_cfg.seed))
    elif name == "time_varying":
        return TimeVaryingDelay([{"start": e.start, "delay": e.delay} for e in delay_cfg.schedule])
    else:
        raise ValueError(f"Unknown delay: {name!r}. Choose from: none, fixed, stochastic, episode, time_varying")
