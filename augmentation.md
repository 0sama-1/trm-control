# Observation Augmentation: Why It Was Added, and What Changed

Context: this repo's delay-aware policies (`conf/delay/*.yaml`) were showing an
odd asymmetry — with `model=dnn`, DDPG converges under a 1-step fixed action
delay; with `model=trm` or `model=transformer`, it doesn't (noisy, no
convergence), even though `trm` already needed noticeably more training time
than `dnn` in the *undelayed* case. This document explains the root cause we
traced it to, the fix (observation augmentation), the diagnostics added
alongside it (mean action delta, deterministic eval rollouts), and how to use
all three. It assumes familiarity with `delay-verification.md` — that document
verified *how much* an action gets delayed; this one is about what the agent
is (and isn't) told about that delay.

## 1. The problem: `DelayedEnv` delays actions but never says so

`DelayedEnv` (`src/delays.py`) queues the agent's actions and, each step,
applies one from `d` steps ago instead of the one the agent just chose:

```python
def step(self, action: Any):
    d = self._delay.sample(self._global_step)
    self._buf.append(action)
    if d == 0:
        applied = action
    elif len(self._buf) > d:
        applied = list(self._buf)[-(d + 1)]   # action chosen d steps ago
    else:
        applied = self.env.action_space.sample()
    ...
    next_obs, reward, ... = self.env.step(applied)
    return next_obs, reward, ...              # <- obs is the raw, un-augmented one
```

That part is correct (verified in `delay-verification.md`). The problem is
what gets *returned*: `next_obs` is the plain Pendulum observation, unchanged.
The agent is never told which of its recent actions is about to land — the
`applied` action lives entirely inside the wrapper's private buffer.

That breaks the assumption every TD-learning algorithm in this repo (DDPG,
TD3, SAC) relies on: that the observation is a sufficient statistic — that
`Q(obs_t, a_t)` is a well-defined, learnable function because `obs_t` fully
explains what `r_t`/`obs_{t+1}` will be, given `a_t`. Under delay, `r_t` and
`obs_{t+1}` are actually produced by `a_{t-d}`, an action decided `d` steps
ago and invisible in `obs_t`. So the exact same `obs_t` can be followed by
different rewards depending on an action the network has no way to see — the
process is no longer Markovian from the agent's point of view, and every
transition `(obs_t, a_t, r_t, obs_{t+1})` written to the replay buffer is
confounded: `a_t` (what got stored) had zero causal role in producing `r_t`
(what it's paired with).

## 2. Why that explains dnn-converges / trm-doesn't, not just "trm is noisier"

The confound's severity scales with how much `a_t` differs from `a_{t-1}`
(more precisely `a_{t-d}`, but adjacent-step is the relevant case at `d=1`).
If consecutive actions are nearly identical, the mislabeled transition is
almost harmless — the critic is still fitting roughly the right thing. If they
differ a lot, the stored `a_t` is essentially an unrelated label on `r_t`.

- `dnn` converges in fewer episodes even *without* delay, so at any matched
  training budget its policy is closer to converged — and a converged
  Pendulum policy is smooth (torque changes gradually between adjacent
  timesteps at a fixed small `dt`). `a_t ≈ a_{t-1}` holds well, so the
  confound is close to a no-op. This is not delay *compensation* — it's the
  confound happening not to bite.
- `trm`'s recursive trunk (`H_cycles=3 × L_cycles=6`, 1-step truncated-backprop
  gradient) is already a harder optimization target, so at the same budget its
  policy is less mature and less smooth. It spends more of the run in the
  regime where `a_t` and `a_{t-1}` diverge — exactly where the confound is
  worst. Its higher capacity is also generally more prone to fitting a noisy
  target than a plain MLP, and the truncated gradient is an already-approximate
  estimator that degrades further under a noisier target.

So the split isn't "trm is too complex" in the abstract — it's that nothing
in the pipeline gives *either* trunk real information about the delay, and
only `dnn` happens to be smooth enough, soon enough, to get away with that.

## 3. The fix: observation (state) augmentation

Standard fix for delayed-action MDPs: append the actions already committed
but not yet applied — the pending queue — to the observation. With
`d`-step delay, the augmented state at decision time `t` is

```
s_t^aug = (obs_t, a_{t-d}, a_{t-d+1}, ..., a_{t-1})
```

`a_{t-d}` — the action that actually determines `r_t`/`obs_{t+1}` — is now
part of what the network conditions on, so `r_t` is fully explained by
`s_t^aug` again and `Q(s_t^aug, a_t)` is well-defined and learnable. The new
action `a_t` doesn't affect this step's reward (still true — its effect is
still `d` steps away), but it does deterministically shape the next augmented
state (the queue shifts: drop `a_{t-d}`, append `a_t`), which is exactly what
makes `(s_t^aug, a_t, r_t, s_{t+1}^aug)` a valid, Markovian transition again.

This also directly targets the second problem discussed alongside the first:
`trm`'s and `transformer`'s trunks were being fed a length-1 vector, so there
was no sequence for recursion/attention to do anything useful with (per
`trm_control.py`'s own docstring). Augmentation turns the input into a real
short sequence (current state + pending actions) — the first point at which
comparing `dnn` vs. `trm` vs. `transformer` under delay actually tests
"can this architecture reason about delay" rather than "which architecture is
easier to optimize on a memoryless vector."

`DelayedEnv` already builds and maintains exactly this queue internally (the
`deque` in `_buf`) to decide which action to apply — augmentation is mostly
*exposing* that existing queue as part of the observation, not new machinery.

## 4. What changed

### `src/delays.py` — `DelayedEnv` gained an `augment_obs` flag

```python
def __init__(self, env: gym.Env, delay: Delay, augment_obs: bool = False):
    ...
    self._augment = augment_obs and self._max_delay > 0
    if self._augment:
        self.observation_space = self._build_augmented_space(env)
```

- `augment_obs=False` (the default) is byte-for-byte the old behavior —
  observation and `observation_space` untouched, so existing configs and
  checkpoints keep working. Verified below.
- `augment_obs=True` extends both `reset()`'s and `step()`'s returned
  observation with `_pending_actions()`: the queue of committed-but-not-yet-
  applied actions, oldest-pending first, zero-padded to `max_delay` slots so
  the shape is constant regardless of how full the buffer is (start of
  episode) or how large the current sampled delay is (stochastic/time-varying
  delay below their max). `observation_space` is widened to match
  (`obs_dim + max_delay * action_dim`), so `obs_dim = env.observation_space.shape[0]`
  — already how every trainer (`ddpg.py`, `td3.py`, `sac.py`, `ppo.py`) builds
  its networks — picks up the new size automatically. No other file needed to
  change to make actor/critic construction aware of the augmented input.
- Automatically a no-op when `delay.name=none` (`max_delay=0` — nothing to
  expose), regardless of `augment_obs`.
- The action-delay *selection* logic (`step()`'s `if d==0 / elif / else`
  block, the part `delay-verification.md` verified) is completely untouched —
  augmentation only changes what gets *returned*, not which action gets
  applied when. Zero risk to the already-verified stochastic-delay fix.

### `conf/config.yaml` — one new toggle

```yaml
eval_every: 10        # run one deterministic (noise-free) eval episode every N training episodes, logged as eval_return
augment_obs: false    # append the pending (committed, not-yet-applied) action queue to the observation. No-op when delay.name=none.
```

Top-level rather than duplicated into each `conf/delay/*.yaml`, since it
applies orthogonally to whichever delay strategy is selected and this keeps
it a single flag to toggle. Turn it on from the CLI same as any other Hydra
override:

```bash
python main.py arch=ddpg model=trm delay=fixed delay.value=1 augment_obs=true
```

Default `false` so nothing about existing runs/scripts changes unless you opt in.

### `src/trainer.py` — env wiring + two new diagnostics

`_make_env` now takes `augment_obs` from config and a `record` flag:

```python
def _make_env(self, record: bool = True) -> gym.Env:
    env = gym.make(self.env_cfg.id, render_mode="rgb_array")
    env = DelayedEnv(env, build_delay(self.cfg.delay), augment_obs=self.cfg.augment_obs)
    if record:
        env = VideoRecorder(...).wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
```

`RLTrainer.__init__` now builds a second instance, `self.eval_env = self._make_env(record=False)`
— same delay + augmentation config as training, just no `RecordVideo` (so it
doesn't collide with the training env's video episode-count/filenames).

`OffPolicyTrainer._training_loop` (DDPG/TD3/SAC) gained:

- **`mean_action_delta`**, logged every episode: `mean(|a_t - a_{t-1}|)` over
  the episode. Directly measures the mechanism from §2 — if `dnn`'s delta
  drops (smooths out) well before `trm`'s does, and the delay=1 return gap
  tracks that gap, that confirms policy smoothness (not raw "complexity") is
  what's driving the difference.
- **`eval_return`**, logged every `cfg.eval_every` episodes: one full episode
  on `eval_env` with exploration noise off (`_select_eval_action`, new
  per-trainer hook — `agent.act(obs)` with no noise arg for DDPG/TD3,
  `agent.act(obs, deterministic=True)` for SAC, which now routes through the
  actor's existing-but-previously-unused mean-of-Gaussian `forward()`).
  Separates policy quality from training-time exploration-noise variance —
  previously every logged `return` was collected *with* OU/Gaussian noise, so
  "noisy return curve" could mean either "unstable policy" or "just
  exploration noise"; there was no way to tell them apart. Now there is.

Scope note: these two additions are in `OffPolicyTrainer` only. `PPOTrainer`
(`OnPolicyTrainer`) has its own rollout-based loop and wasn't part of this
investigation; it still benefits from `augment_obs` transparently (it reads
`obs_dim` off `env.observation_space.shape[0]` the same way), just without the
action-delta/eval-return logging.

### `src/algos/{ddpg,td3,sac}.py` — one new method each

`_select_eval_action` added to each `*Trainer`, plus `SACAgent.act()` gained a
`deterministic: bool = False` param (DDPG's and TD3's `act()` were already
deterministic by default when called with no noise argument, so those needed
no change beyond the new trainer method).

## 5. Known limitation: exact for fixed delay, approximate for stochastic/time-varying

`_pending_actions()` assumes the buffer's oldest entry is always the one that
was just applied, once the buffer is full:

```python
pending = list(self._buf)
if len(pending) > self._max_delay:
    pending = pending[1:]
```

That's *exactly* right for `FixedDelay`/`NoDelay` — `step()`'s `elif` branch
always applies the oldest buffered action once full, since `d` never varies.
It's the case this fix was built and verified for, and the case the current
delay=1..4 experiments use.

For `StochasticDelay`/`TimeVaryingDelay`, whose per-step `d` can be *less*
than `max_delay`, the actually-applied action (`list(self._buf)[-(d+1)]`)
isn't always the oldest one in the buffer, so the exposed "pending queue" is
a reasonable proxy — the `max_delay` most recently committed actions — rather
than a proven sufficient statistic. Extending §3's guarantee to a
resampled-every-step delay isn't a small change (the agent would need to
reason about a delay it can't observe in advance), and wasn't needed for the
comparison at hand. Flagged here rather than silently overclaimed.

## 6. Verification

`torch` isn't needed for any of this (`delays.py`/`trainer.py`'s env wiring
has no torch dependency), so it was checked directly, the same way
`delay-verification.md` checked the delay-selection logic: a minimal
`RecordingEnv` with a distinguishable, known action sequence (`1, 2, 3, ...`),
run through `DelayedEnv` for `FixedDelay(d)`, `d = 0..4`.

**Functional (`augment_obs=True`)**: for every `d`, `observation_space.shape`
matched `obs_dim + d` exactly; the reset observation's action-queue slice was
all zeros; and at every step `t`, the returned pending-queue slice matched the
expected still-in-flight actions (oldest-pending first, zero-padded) computed
independently from the known input sequence — no mismatches across all 5
delay values × 10 steps each.

**Regression (`augment_obs=False`, the default)**: for every `d = 0..4`,
`observation_space.shape` stayed `(1,)` (unchanged), every returned
observation stayed shape `(1,)`, and the applied-action sequence matched
`a_{t-d}` for `t > d` exactly — identical to the pre-change behavior
`delay-verification.md` verified for `FixedDelay`.

## 7. Suggested next experiment

Given §2, a bigger `total_episodes`/`max_steps` run on `trm` alone doesn't
isolate whether the confound or raw optimization budget is the bottleneck —
it conflates both, and even a convergent result wouldn't show `trm` doing
anything a memoryless network couldn't. A tighter comparison at the
**original** budget (200 episodes, 200 max_steps, delay=1):

```bash
# existing baseline — already have this
python main.py arch=ddpg model=trm  delay=fixed delay.value=1 augment_obs=false

# does removing the confound fix it without more compute?
python main.py arch=ddpg model=trm  delay=fixed delay.value=1 augment_obs=true

# the fair comparison: same input info, does trm's recursion do anything dnn can't?
python main.py arch=ddpg model=dnn  delay=fixed delay.value=1 augment_obs=true
```

Read `mean_action_delta` and `eval_return` alongside `return`/`avg100_return`
on all three — they're what turns "did the number go up" into "did the
mechanism in §2 actually resolve."
