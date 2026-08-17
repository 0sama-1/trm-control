# Delay Mechanism: How It Works, and Verification of All Four Strategies

Context: a review of `src/delays.py` (the action-delay simulation used to
train delay-robust control policies), prompted by a suspicion that the
default (`stochastic`) delay "does not seem correct." It wasn't. This
document explains how the mechanism works end to end, then walks through
the empirical verification performed on all four `Delay` strategies —
`none`, `fixed`, `stochastic`, `time_varying` — including the bug found and
fixed in `stochastic`.

## 1. Gymnasium has no native concept of action delay

`gym.Env`'s API is synchronous and immediate: `obs, info = env.reset()`,
then `obs, reward, terminated, truncated, info = env.step(action)`.
Whatever `action` is passed to `step()` is applied to the simulator right
then — there's no built-in hook for "apply the action I sent N steps ago
instead."

To simulate delay, this repo uses Gymnasium's standard extension point:
`gym.Wrapper`. A wrapper subclass sits between the training loop and the
real env, overrides `step`/`reset`, and — as long as it eventually calls
`self.env.step(...)` and returns a same-shaped tuple — is fully transparent
to everything downstream (other wrappers, `RecordEpisodeStatistics`, the
training loop). That's `DelayedEnv`.

## 2. `DelayedEnv` — the wrapper that actually delays actions

```python
# src/delays.py
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
```

Each call to `step()`:

1. Asks the `Delay` strategy how many steps to hold the action back, *this*
   call (`d = self._delay.sample(self._global_step)`).
2. Buffers the agent's newest action.
3. Looks `d` entries back in the buffer and applies *that* action to the
   real env instead of the current one — `list(self._buf)[-(d + 1)]` is the
   action taken exactly `d` steps ago (current action is index `-1`, one
   step back is `-2`, ..., `d` steps back is `-(d+1)`).
4. Early in an episode, before `d` actions have accumulated, there's
   nothing valid to look up yet — falls back to a random action rather than
   under-indexing or crashing.

The buffer is sized `max_delay + 1` — exactly enough capacity to always
hold the oldest action any valid `d` could reach back to, no more. **This
sizing is an invariant every `Delay.sample()` must respect: it must never
return more than `max_delay`.** That invariant is what turned out to be
broken (§5.3).

`DelayedEnv` is spliced in first, closest to the raw simulator, in
`trainer.py::_make_env`, so everything recorded/logged downstream
(video, `RecordEpisodeStatistics`, wandb) sees the *delayed* trajectory the
agent actually experienced:

```python
env = gym.make(self.env_cfg.id, render_mode="rgb_array")
env = DelayedEnv(env, build_delay(self.cfg.delay))
env = VideoRecorder(self.cfg.video_folder, self.cfg.record_every, self.arch.name).wrap(env)
env = gym.wrappers.RecordEpisodeStatistics(env)
```

## 3. The `Delay` strategy hierarchy

"How much to delay" is decoupled from "how delay gets applied to the env."
`Delay` (ABC) declares two members every strategy must supply:

```python
class Delay(ABC):
    @abstractmethod
    def sample(self, total_steps: int) -> int:
        """Return the number of environment steps to delay the action."""

    @property
    @abstractmethod
    def max_delay(self) -> int:
        """Upper bound on the delay, used to size the action history buffer."""
```

Four implementations, selected via `conf/delay/*.yaml` + `build_delay()`:

| Strategy | `conf/delay/*.yaml` | Behavior |
|---|---|---|
| `NoDelay` | `none.yaml` | Always `0`. |
| `FixedDelay` | `fixed.yaml` | Constant `value` for the whole run. |
| `StochasticDelay` | `stochastic.yaml` | Resampled from a distribution every step. |
| `TimeVaryingDelay` | `time_varying.yaml` | Piecewise-constant over *total training steps* (a curriculum, e.g. delay `0 → 1 → 3 → 0`). |

`DelayedEnv` never knows which one it's holding — it just calls `.sample()`
and reads `.max_delay` once, at construction.

## 4. Verification method

No GPU/gym-training environment available here, so each strategy was
verified in a disposable CPU-only venv (`numpy` + `gymnasium` +
`omegaconf`, no `torch` needed — `delays.py` has no torch dependency):

- A minimal `RecordingEnv(gym.Env)` that does nothing but log whatever
  action it's actually handed via `step()`. Wrapping it in `DelayedEnv` and
  feeding it a known, distinguishable action sequence (`0, 1, 2, ...`)
  makes it possible to check, exactly, which action got applied at every
  timestep.
- Direct `Delay.sample()` sampling in bulk (10k–20k draws) to check
  distributional bounds.
- `build_delay()` round-tripped against every real `conf/delay/*.yaml` file
  to catch config/code mismatches, not just unit-level logic.

## 5. Findings, per strategy

### 5.1 `NoDelay` — verified correct

```python
class NoDelay(Delay):
    def sample(self, total_steps: int) -> int:
        return 0
    @property
    def max_delay(self) -> int:
        return 0
```

Trivial by construction (buffer size 1, `d` always `0` triggers `DelayedEnv`'s
immediate-apply shortcut). Confirmed empirically: applied-action log ==
chosen-action log, exactly, for every step.

### 5.2 `FixedDelay` — verified correct

```python
class FixedDelay(Delay):
    def __init__(self, value: int):
        self._value = value
    def sample(self, total_steps: int) -> int:
        return self._value
    @property
    def max_delay(self) -> int:
        return self._value
```

`sample()` ignoring `total_steps` and always returning `self._value` is the
*intended* contract — "fixed" means a constant delay for the whole run, as
opposed to `stochastic` (resamples every call) or `time_varying` (changes
with total training steps).

Traced end-to-end for `value = 0, 1, 3` against `RecordingEnv`:

- The first `value` steps of each episode get the random-fallback action
  (buffer not full yet — expected warm-up).
- From step `value` onward, the applied action is **exactly** the one the
  agent chose `value` steps earlier, every time, with no drift.
- `value = 0` degenerates cleanly to no delay at all.
- `conf/delay/fixed.yaml` (`value: 1.0`) round-trips through `build_delay()`
  to `FixedDelay(1)` with `sample() == 1`, `max_delay == 1` (already an
  `int` — `build_delay` casts via `int(delay_cfg.value)` before
  construction, so no stray-float `maxlen` issue reaches `deque()`).

One latent (not live) nitpick: `FixedDelay.__init__` doesn't itself enforce
`value` is an `int`. If constructed directly with a float (bypassing
`build_delay`), `max_delay` would return a float and
`deque(maxlen=delay.max_delay + 1)` would raise `TypeError`. Every actual
call site in this repo goes through `build_delay`'s cast, so this isn't
reachable today.

### 5.3 `StochasticDelay` — **bug found and fixed**

Two independent problems, both in the direction of "delay silently degrades
into feeding the environment random actions instead of the agent's own":

#### 5.3.1 Config values out of scale (the practical, default-config bug)

`conf/delay/stochastic.yaml` — the *default* delay
(`conf/config.yaml`'s `defaults:` picks `delay: stochastic` unless
overridden) — had:

```yaml
distribution: uniform
min: 10.0
max: 500.0
```

Compare to the sibling configs: `fixed.yaml` uses `value: 1`,
`time_varying.yaml` schedules delays of `0/1/3/0`. A 10–500 step delay is
1–2 orders of magnitude larger than those, and larger than the episodes
themselves (`Pendulum-v1`: `max_steps=200`, `CartPole-v1`: `max_steps=500`).

Consequence, in `DelayedEnv.step`: whenever the sampled delay `d` exceeds
how much action history has accumulated so far in the episode, the applied
action falls back to `self.env.action_space.sample()` — a **uniformly
random action**, discarding the agent's chosen action entirely (not merely
delaying it). Simulated against the old config:

| Env | `max_steps` | Fraction of steps that got a random action instead of the agent's |
|---|---|---|
| Pendulum-v1 | 200 | **81.5%** |
| CartPole-v1 | 500 | **50.9%** |

i.e. training with the *default* delay config was mostly training against
random actions, not delayed ones — no policy can learn well under that.

#### 5.3.2 `normal` / `exponential` never clipped to `max_val`

```python
elif self._dist == "normal":
    mean = (self._min + self._max) / 2.0
    std = (self._max - self._min) / 6.0
    return max(0, int(round(self._rng.normal(mean, std))))   # only floors at 0, no upper clip
elif self._dist == "exponential":
    return max(0, int(round(self._rng.exponential(scale=self._min))))  # self._max unused entirely
```

`DelayedEnv`'s buffer is sized on the assumption every sampled delay is
`<= max_delay` (§2). Neither `normal` nor `exponential` enforced that upper
bound: a sample above `max_delay` can never be satisfied by the buffer
(`len(self._buf) > d` becomes permanently false for the rest of that
episode) — silently triggering the same random-action fallback as §5.3.1,
indefinitely, with no error or warning. `exponential` additionally never
used `max_val` for sampling at all (only repurposed `min_val` as its scale
parameter), making "max" dead config for that branch.

A minor doc/code mismatch rode along with this: the yaml comment said
`# or normal, exponential, lognormal`, but `lognormal` was never
implemented — selecting it raised `ValueError`.

#### 5.3.3 Fix applied

`src/delays.py`:

```python
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
```

All three distributions now share one clip-to-`[min, max]` step before
returning, so the `max_delay` buffer-sizing invariant holds
unconditionally, for every distribution — no more silent degradation for
`normal`/`exponential`. The unknown-distribution error now also lists valid
choices, matching the style used elsewhere in the codebase (`dnn.py`'s
`Unknown activation: ...`, `main.py`'s `Unknown algorithm: ...`).

`conf/delay/stochastic.yaml`:

```yaml
name: stochastic
distribution: uniform  # or normal, exponential
min: 0.0
max: 5.0
seed: ${seed}
```

Rescaled to a range consistent with the other delay configs (`lognormal`
also dropped from the comment, since it was never implemented).

#### 5.3.4 Verification of the fix

- **Invariant check**: 20,000 samples drawn per distribution, at several
  `min`/`max` pairs (including the old `10/500` values), for
  `uniform`/`normal`/`exponential`. Every sample landed within
  `[min, max_delay]` — confirmed for all three, including the two that were
  previously unbounded.
- **Unknown-distribution error**: `StochasticDelay(distribution="lognormal", ...)`
  raises `ValueError: Unknown distribution: 'lognormal'. Choose from: uniform, normal, exponential`.
- **Fallback-rate re-simulation**, against the new `stochastic.yaml`
  through `build_delay()`:

  | Env | Before fix | After fix |
  |---|---|---|
  | Pendulum-v1 (200 steps) | 81.5% | **1.25%** |
  | CartPole-v1 (500 steps) | 50.9% | **0.50%** |

  The small residual left after the fix is exactly the intended
  start-of-episode warm-up before the history buffer fills — matching
  `DelayedEnv`'s own docstring ("If not enough history exists yet, a random
  action is used instead"), not a leftover bug.

### 5.4 `TimeVaryingDelay` — verified correct

```python
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
```

Since `d` can *change* between calls (unlike `Fixed`/`Stochastic`, which are
i.i.d. per call or constant), the interesting question is whether the fixed
buffer capacity (`max_delay + 1`, sized once at construction from the
schedule's largest delay) still yields correct lookups as `d` moves around
within that ceiling. It does — `list(self._buf)[-(d + 1)]` only ever looks
back `d` steps into whatever is currently buffered, and `d` is guaranteed
`<= max_delay` by construction (it's drawn from the same `schedule` that
`max_delay` maxes over), so the buffer always has enough history once
warmed up, regardless of how `d` has moved.

Verified with a schedule `{0: 0, 5: 2, 10: 4}` over 20 global steps: the
per-step delay sequence resolved to `[0]*5 + [2]*5 + [4]*10` as expected,
and every post-warm-up step's applied action matched the action chosen
exactly `d` steps earlier — including the transition steps where `d`
itself just changed. `conf/delay/time_varying.yaml`'s real schedule
(`0→0, 10000→1, 20000→3, 30000→0`) round-tripped correctly too:
`sample(0)=0`, `sample(15000)=1`, `sample(25000)=3`, `sample(35000)=0`,
`max_delay=3`.

## 6. Summary

| Strategy | Verdict |
|---|---|
| `none` | Correct |
| `fixed` | Correct |
| `stochastic` | **Was broken** (config scale + missing clip on `normal`/`exponential`) — fixed, re-verified |
| `time_varying` | Correct |

The only strategy actually exercised by default training runs before this
fix was `stochastic` (`conf/config.yaml`'s default `delay:` group) — which
is exactly the one that was broken, and by a wide enough margin (81.5%
random-action rate on Pendulum) to plausibly explain any "delay training
doesn't seem to work" observations up to this point.
