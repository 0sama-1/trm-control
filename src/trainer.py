from __future__ import annotations

import json
import platform
import socket
import time
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from delays import DelayedEnv, PrevActionObs, build_delay
from logger import WandbLogger
from recorder import VideoRecorder
from utils import get_device, set_seed


def _count_params(obj) -> int:
    """Sum parameters of every nn.Module attribute on `obj`, excluding
    target networks (copies of the trainable ones, not distinct capacity).
    Works unmodified across DDPG/TD3/SAC (which hold actor/critic on a
    separate `self.agent`) and PPO (which holds them directly on the
    trainer) since the caller passes `getattr(self, "agent", self)` --
    no per-algo special-casing needed. See _write_results_summary — first-
    class metric per
    projects/trm-delay-robust-scaling/ideas/proposal-final.json."""
    total = 0
    for name, val in vars(obj).items():
        if name.startswith("target"):
            continue
        if isinstance(val, torch.nn.Module):
            total += sum(p.numel() for p in val.parameters())
    return total


class RLTrainer(ABC):
    """Base class for all RL training pipelines.

    Handles: device/seed setup, folder creation, env construction (delay +
    optional obs augmentation + video recording + episode stats, plus a
    matching no-video eval_env instance), wandb lifecycle, and the top-level
    train() skeleton.

    Subclasses implement: _build_agent, _training_loop, _save_checkpoint,
    _load_checkpoint.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.arch = cfg.arch
        self.env_cfg = cfg.env
        self.device = get_device()
        set_seed(cfg.seed)

        Path(cfg.model_folder).mkdir(parents=True, exist_ok=True)
        Path(cfg.video_folder).mkdir(parents=True, exist_ok=True)
        # Same identity fields/order as the wandb run name (logger.py), minus the
        # timestamp — the checkpoint path must stay stable across runs to resume.
        with open_dict(cfg):
            cfg.model_id = str(uuid4())
            cfg.video_folder = f"{cfg.video_folder}/{cfg.model_id}"
            Path(cfg.video_folder).mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"{cfg.model_id}.pt"
        self.model_path = str(Path(cfg.model_folder) / checkpoint_name)

        self.env = self._make_env()
        # Separate instance for deterministic eval rollouts (_evaluate, OffPolicyTrainer)
        # so they don't perturb the training env's DelayedEnv buffer/RecordVideo episode
        # count. Same delay + augmentation config, no video.
        self.eval_env = self._make_env(record=False)
        # Warmup actions and DelayedEnv's start-of-episode fill both come from
        # action_space.sample(); seed it so a run is reproducible end to end.
        self.env.action_space.seed(cfg.seed)
        self.eval_env.action_space.seed(cfg.seed + 1)
        self.logger = WandbLogger(cfg, self.arch.name)

        self.episode_count: int = 0
        self.total_steps: int = 0
        self._recent_returns: list[float] = []
        self._start_time = time.time()
        self._last_avg100: float = 0.0
        self._episode_history: list[dict] = []  # one row per _record_return call
                                                  # -- see _write_results_summary
        self._step_history: list[dict] = []      # one row per env step, when
                                                   # cfg.track_step_history (default
                                                   # true) -- see OffPolicyTrainer
                                                   # ._training_loop

    # Common set ups

    def _make_env(self, record: bool = True) -> gym.Env:
        env = gym.make(self.env_cfg.id, render_mode="rgb_array")
        env = DelayedEnv(
            env,
            build_delay(self.cfg.delay),
            augment_obs=self.cfg.augment_obs,
            expose_realised=bool(self.cfg.delay.get("expose_realised", False)),
        )
        if self.cfg.get("prev_action_obs", False):
            env = PrevActionObs(env)
        if record:
            env = VideoRecorder(self.cfg.video_folder, self.cfg.record_every, self.arch.name).wrap(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def _record_return(self, ep_return: float, extra: dict | None = None) -> float:
        """Update the rolling return history, log to wandb, return avg100."""
        self._recent_returns.append(ep_return)
        if len(self._recent_returns) > 100:
            self._recent_returns.pop(0)
        avg100 = float(np.mean(self._recent_returns))
        self._last_avg100 = avg100
        row = {
            "episode": self.episode_count,
            "return": ep_return,
            "avg100_return": avg100,
            "total_steps": self.total_steps,
            **(extra or {}),
        }
        # Keep every row for the results-summary history (small: one dict of
        # scalars per episode, not per step) so a rerun's full learning curve
        # -- not just the final number -- is inspectable after the fact
        # without needing W&B; see _write_results_summary.
        self._episode_history.append(row)
        self.logger.log(row)
        return avg100

    # Abstract methods

    @abstractmethod
    def _build_agent(self) -> None:
        """Instantiate networks, optimisers, buffers, and any other state."""
        ...

    @abstractmethod
    def _training_loop(self) -> None:
        """Run the full training loop (episodes or rollouts)."""
        ...

    @abstractmethod
    def _save_checkpoint(self) -> None:
        """Persist model weights and episode_count to self.model_path."""
        ...

    @abstractmethod
    def _load_checkpoint(self) -> int:
        """Load weights from self.model_path, return episode_count to resume from."""
        ...

    # Entry point

    def _write_results_summary(self) -> None:
        """Write final reward, wall-clock time, parameter count, AND the
        full per-episode history (return, avg100, eval_return when logged,
        mean_action_delta, algo-specific losses) to
        results/<env>_<arch>_<model>_<delay>_seed<seed>.json -- first-class
        metrics per projects/trm-delay-robust-scaling/ideas/proposal-final.json,
        not an afterthought (research.boundaries). One file per run, named so
        concurrent PBS jobs never collide. The history array is what makes a
        run's learning curve (not just its final number) inspectable after
        the fact -- e.g. to tell "converged early and plateaued" from "still
        improving when total_episodes ran out" -- without needing W&B, and
        is also what a rerun uses to diff against a previous run's curve."""
        results_folder = Path(getattr(self.cfg, "results_folder", "results"))
        results_folder.mkdir(parents=True, exist_ok=True)
        # augment_obs is included whenever it can vary independently of
        # delay (Phase 2's augmentation ablation runs both aug=true and
        # aug=false at the SAME delay condition) -- without this, two
        # distinct runs collide on one filename and one silently
        # overwrites the other. Omitted for delay=none, where augment_obs
        # is a no-op (max_delay=0) and was never varied there, to keep
        # Phase 1's existing filenames stable.
        aug_tag = "" if self.cfg.delay.name == "none" else f"_aug{int(bool(self.cfg.augment_obs))}"
        # run_tag separates Phase 3 arms that share every other identity field
        # (e.g. the delay-oracle arm vs. plain stochastic delay); empty by
        # default so every earlier filename is unchanged.
        run_tag = str(self.cfg.get("run_tag", "") or "")
        tag = f"_{run_tag}" if run_tag else ""
        name = (
            f"{self.env_cfg.id}_{self.arch.name}_{self.cfg.model.name}"
            f"_{self.cfg.delay.name}{aug_tag}{tag}_seed{self.cfg.seed}.json"
        )
        summary = {
            "env_id": self.env_cfg.id,
            "arch": self.arch.name,
            "model": self.cfg.model.name,
            "delay": self.cfg.delay.name,
            "delay_value": getattr(self.cfg.delay, "value", None),
            "seed": self.cfg.seed,
            "augment_obs": bool(self.cfg.augment_obs),
            "total_episodes": self.arch.total_episodes,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "final_avg100_return": self._last_avg100,
            "wall_clock_seconds": time.time() - self._start_time,
            "param_count": _count_params(getattr(self, "agent", self)),
            "model_id": self.cfg.model_id,
            "run_tag": run_tag,
            "device": self.device,
            "device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else platform.processor(),
            "hostname": socket.gethostname(),
            # Full resolved config, so a checkpoint can be rebuilt exactly
            # from its results file (Phase 3 evaluation harness, src/eval.py).
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "wandb_enabled": bool(self.cfg.wandb.enabled),
            "wandb_run_path": self.logger.run_path,
            "episode_history": self._episode_history,
            "step_history": self._step_history,
        }
        with open(results_folder / name, "w") as f:
            json.dump(summary, f, indent=2)

    def train(self) -> None:
        self._build_agent()
        if Path(self.model_path).exists():
            self.episode_count = self._load_checkpoint()
            print(f"Loaded checkpoint from episode {self.episode_count}")
        self._training_loop()
        # TODO: add a hydra config to save model (or not)?
        self._save_checkpoint()
        self._write_results_summary()
        self.env.close()
        self.logger.finish()
        print(f"\nTraining complete. Model: {self.model_path} | Videos: {self.cfg.video_folder}/")


class OffPolicyTrainer(RLTrainer):
    """Episode-based training loop for off-policy algorithms (DDPG, TD3, SAC).

    Each episode: reset → collect steps → update buffer → gradient steps.
    Also runs a periodic noise-free eval rollout (_evaluate, every cfg.eval_every
    episodes) and tracks mean per-episode action delta — both diagnostics for the
    delay confound, see augmentation.md. Subclasses implement the agent-specific
    pieces via six abstract methods.
    """

    def _episodes(self):
        """Episode indices to run. Default: up to arch.total_episodes. With
        arch.total_steps set (Phase 4, env-step budget for environments whose
        episode length varies with skill, e.g. InvertedDoublePendulum), no new
        episode starts once the budget is reached; the last one runs to its end."""
        budget = self.arch.get("total_steps")
        if budget is None:
            yield from range(self.episode_count, self.arch.total_episodes)
            return
        ep = self.episode_count
        while self.total_steps < budget:
            yield ep
            ep += 1

    def _training_loop(self) -> None:
        track_steps = bool(getattr(self.cfg, "track_step_history", True))
        for ep in self._episodes():
            obs, _ = self.env.reset(seed=self.cfg.seed + ep)
            self._on_episode_start()

            ep_return = 0.0
            prev_action: np.ndarray | None = None
            action_deltas: list[float] = []
            step_in_episode = 0
            for _ in range(self.env_cfg.max_steps):
                action = self._select_action(obs)
                if prev_action is not None:
                    action_deltas.append(float(np.mean(np.abs(action - prev_action))))
                prev_action = action

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self._store_transition(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_return += float(reward)
                self.total_steps += 1
                self._maybe_update()
                if track_steps:
                    # Per-step record: the actually-applied (post-delay) action
                    # and sampled delay are the direct diagnostic for this
                    # project's central premise (the delay confound, see
                    # augmentation.md) -- previously invisible outside the
                    # wrapper's private buffer. Scoped to env-level signals
                    # only (action/applied_action/reward/delay), not
                    # per-step actor/critic loss, which would require
                    # touching every algo file (ddpg/td3/sac) individually;
                    # those stay at episode granularity (actor_loss/
                    # critic_loss in episode_history).
                    step_row = {
                        "global_step": self.total_steps,
                        "episode": self.episode_count,
                        "step_in_episode": step_in_episode,
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "applied_action": info.get("delay_applied_action"),
                        "delay_sampled_steps": info.get("delay_sampled_steps"),
                        "reward": float(reward),
                        "done": bool(done),
                    }
                    self._step_history.append(step_row)
                    if self.cfg.wandb.enabled and self.cfg.wandb.log_steps:
                        # Separate x-axis ("global_step") from the per-episode
                        # log's ("episode"/"total_steps") so W&B doesn't
                        # collapse the two into one step counter.
                        self.logger.log_step(step_row)
                step_in_episode += 1
                if done:
                    break

            self.episode_count += 1
            metrics = self._episode_metrics()
            # Diagnostic for the delay confound (see augmentation.md): how much a_t
            # differs from a_{t-1} step to step. A large value means the delay=d
            # transition stored this episode (paired with a_{t-d}, not a_t) is a
            # poor proxy for a_t's real effect.
            metrics["mean_action_delta"] = float(np.mean(action_deltas)) if action_deltas else 0.0
            if self.episode_count % self.cfg.eval_every == 0:
                # Noise-free rollout, logged separately from the exploration-noisy
                # training return above, so the two aren't conflated when reading
                # the curves (see augmentation.md).
                metrics["eval_return"] = self._evaluate()
            avg100 = self._record_return(ep_return, metrics)

            if self.episode_count % self.cfg.print_stat_every == 0:
                self._print_stats(ep_return, avg100, metrics)

            if self.episode_count % self.cfg.record_every == 0:
                self._save_checkpoint()

    def _evaluate(self) -> float:
        """Run one deterministic (noise-free) episode on the held-out eval env and
        return its total reward. Uses the same delay + augmentation config as
        training, just no exploration noise on the action."""
        obs, _ = self.eval_env.reset(seed=self.cfg.seed + 1_000_000 + self.episode_count)
        self._on_eval_start()
        ep_return = 0.0
        for _ in range(self.env_cfg.max_steps):
            action = self._select_eval_action(obs)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            ep_return += float(reward)
            if terminated or truncated:
                break
        return ep_return

    def _on_eval_start(self) -> None:
        """Called before each deterministic eval episode (e.g. reset a carried latent)."""

    @abstractmethod
    def _on_episode_start(self) -> None:
        """Called at the start of each episode (e.g. noise reset, loss list clear)."""
        ...

    @abstractmethod
    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        """Return an action for the current obs (handles warmup / noise internally)."""
        ...

    @abstractmethod
    def _select_eval_action(self, obs: np.ndarray) -> np.ndarray:
        """Return a noise-free (deterministic/greedy) action, for _evaluate()."""
        ...

    @abstractmethod
    def _store_transition(self, obs, action, reward, next_obs, done) -> None:
        """Add a transition to the replay buffer."""
        ...

    @abstractmethod
    def _maybe_update(self) -> None:
        """Run gradient updates if the buffer is ready (called every step)."""
        ...

    @abstractmethod
    def _episode_metrics(self) -> dict:
        """Return algo-specific per-episode metrics (actor_loss, critic_loss, …)."""
        ...

    def _print_stats(self, ep_return: float, avg100: float, metrics: dict) -> None:
        print(
            f"Episode {self.episode_count:4d}/{self.arch.total_episodes} | "
            f"Return: {ep_return:8.2f} | Avg100: {avg100:8.2f}"
        )


class OnPolicyTrainer(RLTrainer):
    """Rollout-based training loop for on-policy algorithms (PPO).

    Collects a fixed number of steps per rollout; episodes emerge from done
    flags mid-rollout. Subclasses implement rollout collection, policy update,
    and update-level metric reporting.
    """

    def _training_loop(self) -> None:
        obs = self._initial_obs()
        while self.episode_count < self.arch.total_episodes:
            obs = self._collect_rollout(obs)
            self._update_policy()
            self.logger.log({**self._update_metrics(), "total_steps": self.total_steps})

    @abstractmethod
    def _initial_obs(self):
        """Return the initial observation (may need tensor conversion)."""
        ...

    @abstractmethod
    def _collect_rollout(self, obs):
        """Collect one rollout of experience.

        Must update self.episode_count, self.total_steps, and call
        self._record_return() for each completed episode.
        Returns the final observation for the next rollout.
        """
        ...

    @abstractmethod
    def _update_policy(self) -> None:
        """Run the policy/value update over the collected rollout."""
        ...

    @abstractmethod
    def _update_metrics(self) -> dict:
        """Return update-level metrics (pg_loss, value_loss, entropy, …)."""
        ...
