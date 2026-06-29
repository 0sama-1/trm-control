import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from hydra.utils import instantiate

from models.dnn import make_head
from trainer import OffPolicyTrainer
from utils import soft_update


# ---------------------------------------------------------------------------
# Model / agent components
# ---------------------------------------------------------------------------

class OUNoise:
    def __init__(self, action_dim: int, theta: float, sigma: float, dt: float):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.zeros(action_dim, dtype=np.float32)

    def reset(self):
        self.state.fill(0.0)

    def sample(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (-x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state.astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return (
            np.asarray(s,  dtype=np.float32),
            np.asarray(a,  dtype=np.float32),
            np.asarray(r,  dtype=np.float32).reshape(-1, 1),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d,  dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, model_cfg: DictConfig):
        super().__init__()
        self.action_limit = action_limit
        self.trunk = instantiate(model_cfg, in_dim=obs_dim)
        self.head  = make_head(self.trunk.out_dim, act_dim, model_cfg.output_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.trunk(obs))) * self.action_limit


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, model_cfg: DictConfig):
        super().__init__()
        self.trunk = instantiate(model_cfg, in_dim=obs_dim + act_dim)
        self.head  = make_head(self.trunk.out_dim, 1, model_cfg.output_init)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(torch.cat([obs, act], dim=-1)))


class DDPGAgent:
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, arch_cfg: DictConfig, model_cfg: DictConfig, device: str):
        self.device = device
        self.action_limit = action_limit
        self.cfg = arch_cfg

        self.actor  = Actor(obs_dim, act_dim, action_limit, model_cfg).to(device)
        self.critic = Critic(obs_dim, act_dim, model_cfg).to(device)
        self.target_actor  = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=arch_cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=arch_cfg.critic_lr)

    def act(self, obs: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if noise is not None:
            action = action + noise
        return np.clip(action, -self.action_limit, self.action_limit)

    def update(self, replay: ReplayBuffer) -> tuple[float, float]:
        cfg = self.cfg
        s, a, r, ns, d = replay.sample(cfg.batch_size)
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.float32, device=self.device)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_a   = self.target_actor(ns)
            target_q = self.target_critic(ns, next_a)
            y = r + cfg.gamma * (1.0 - d) * target_q

        critic_loss = nn.functional.mse_loss(self.critic(s, a), y)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        soft_update(self.target_actor,  self.actor,  cfg.tau)
        soft_update(self.target_critic, self.critic, cfg.tau)

        return actor_loss.item(), critic_loss.item()

    def save(self, path: str, episode_count: int):
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic":self.target_critic.state_dict(),
            "actor_opt":    self.actor_opt.state_dict(),
            "critic_opt":   self.critic_opt.state_dict(),
            "episode_count":episode_count,
        }, path)

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        return ckpt.get("episode_count", 0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DDPGTrainer(OffPolicyTrainer):
    def _build_agent(self) -> None:
        obs_dim      = self.env.observation_space.shape[0]
        act_dim      = self.env.action_space.shape[0]
        action_limit = float(self.env.action_space.high[0])

        self.agent  = DDPGAgent(obs_dim, act_dim, action_limit, self.arch, self.cfg.model, self.device)
        self.replay = ReplayBuffer(self.arch.buffer_size)
        self.noise  = OUNoise(act_dim, self.arch.ou_noise.theta, self.arch.ou_noise.sigma, self.arch.ou_noise.dt)
        self._action_limit = action_limit
        self._actor_losses:  list[float] = []
        self._critic_losses: list[float] = []

    def _on_episode_start(self) -> None:
        self.noise.reset()
        self._actor_losses  = []
        self._critic_losses = []

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        if self.total_steps < self.arch.warmup_steps:
            return self.env.action_space.sample()
        return self.agent.act(obs, noise=self.noise.sample() * self._action_limit)

    def _store_transition(self, obs, action, reward, next_obs, done) -> None:
        self.replay.add(obs, action, reward, next_obs, done)

    def _maybe_update(self) -> None:
        if len(self.replay) >= self.arch.batch_size:
            for _ in range(self.arch.updates_per_step):
                a_loss, c_loss = self.agent.update(self.replay)
                self._actor_losses.append(a_loss)
                self._critic_losses.append(c_loss)

    def _episode_metrics(self) -> dict:
        return {
            "actor_loss":  float(np.mean(self._actor_losses))  if self._actor_losses  else 0.0,
            "critic_loss": float(np.mean(self._critic_losses)) if self._critic_losses else 0.0,
            "buffer_size": len(self.replay),
        }

    def _print_stats(self, ep_return: float, avg100: float, metrics: dict) -> None:
        print(
            f"Episode {self.episode_count:4d}/{self.arch.total_episodes} | "
            f"Return: {ep_return:8.2f} | Avg100: {avg100:8.2f} | "
            f"ActorLoss: {metrics['actor_loss']:9.4f} | CriticLoss: {metrics['critic_loss']:9.4f}"
        )

    def _save_checkpoint(self) -> None:
        self.agent.save(self.model_path, self.episode_count)

    def _load_checkpoint(self) -> int:
        return self.agent.load(self.model_path)
