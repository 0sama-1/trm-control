from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, open_dict

from delays import DelayedEnv, build_delay
from logger import WandbLogger
from recorder import VideoRecorder
from utils import get_device, set_seed


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
        self.logger = WandbLogger(cfg, self.arch.name)

        self.episode_count: int = 0
        self.total_steps: int = 0
        self._recent_returns: list[float] = []

    # Common set ups

    def _make_env(self, record: bool = True) -> gym.Env:
        env = gym.make(self.env_cfg.id, render_mode="rgb_array")
        env = DelayedEnv(env, build_delay(self.cfg.delay), augment_obs=self.cfg.augment_obs)
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
        self.logger.log({
            "episode": self.episode_count,
            "return": ep_return,
            "avg100_return": avg100,
            "total_steps": self.total_steps,
            **(extra or {}),
        })
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

    def train(self) -> None:
        self._build_agent()
        if Path(self.model_path).exists():
            self.episode_count = self._load_checkpoint()
            print(f"Loaded checkpoint from episode {self.episode_count}")
        self._training_loop()
        # TODO: add a hydra config to save model (or not)?
        self._save_checkpoint()
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

    def _training_loop(self) -> None:
        for ep in range(self.episode_count, self.arch.total_episodes):
            obs, _ = self.env.reset(seed=self.cfg.seed + ep)
            self._on_episode_start()

            ep_return = 0.0
            prev_action: np.ndarray | None = None
            action_deltas: list[float] = []
            for _ in range(self.env_cfg.max_steps):
                action = self._select_action(obs)
                if prev_action is not None:
                    action_deltas.append(float(np.mean(np.abs(action - prev_action))))
                prev_action = action

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self._store_transition(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_return += float(reward)
                self.total_steps += 1
                self._maybe_update()
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
        ep_return = 0.0
        for _ in range(self.env_cfg.max_steps):
            action = self._select_eval_action(obs)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            ep_return += float(reward)
            if terminated or truncated:
                break
        return ep_return

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
