import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from trainer import OffPolicyTrainer
from utils import fanin_init, soft_update


# ---------------------------------------------------------------------------
# Model / agent components
# ---------------------------------------------------------------------------

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
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, hidden_dim: int):
        super().__init__()
        self.action_limit = action_limit
        self.net = nn.Sequential(
            fanin_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            fanin_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias,   -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.action_limit


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            fanin_init(nn.Linear(obs_dim + act_dim, hidden_dim)),
            nn.ReLU(),
            fanin_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias,   -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class TD3Agent:
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, arch_cfg: DictConfig, device: str):
        self.device = device
        self.action_limit = action_limit
        self.cfg = arch_cfg
        self._critic_update_count = 0

        h = arch_cfg.hidden_dim
        self.actor   = Actor(obs_dim, act_dim, action_limit, h).to(device)
        self.critic1 = Critic(obs_dim, act_dim, h).to(device)
        self.critic2 = Critic(obs_dim, act_dim, h).to(device)

        self.target_actor   = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=arch_cfg.actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=arch_cfg.critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=arch_cfg.critic_lr)

    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if noise_std > 0.0:
            action += np.random.randn(*action.shape).astype(np.float32) * noise_std
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
            noise  = (torch.randn_like(a) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_a = (self.target_actor(ns) + noise).clamp(-self.action_limit, self.action_limit)
            target_q = torch.min(self.target_critic1(ns, next_a), self.target_critic2(ns, next_a))
            y = r + cfg.gamma * (1.0 - d) * target_q

        c1_loss = nn.functional.mse_loss(self.critic1(s, a), y)
        self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()

        c2_loss = nn.functional.mse_loss(self.critic2(s, a), y)
        self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()

        self._critic_update_count += 1
        actor_loss = 0.0

        if self._critic_update_count % cfg.policy_delay == 0:
            actor_loss = -self.critic1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            actor_loss = actor_loss.item()
            soft_update(self.target_actor,   self.actor,   cfg.tau)
            soft_update(self.target_critic1, self.critic1, cfg.tau)
            soft_update(self.target_critic2, self.critic2, cfg.tau)

        return actor_loss, c1_loss.item()

    def save(self, path: str, episode_count: int):
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "target_actor":   self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_opt":      self.actor_opt.state_dict(),
            "critic1_opt":    self.critic1_opt.state_dict(),
            "critic2_opt":    self.critic2_opt.state_dict(),
            "episode_count":  episode_count,
        }, path)

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic1.load_state_dict(ckpt["target_critic1"])
        self.target_critic2.load_state_dict(ckpt["target_critic2"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
        self.critic2_opt.load_state_dict(ckpt["critic2_opt"])
        return ckpt.get("episode_count", 0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TD3Trainer(OffPolicyTrainer):
    def _build_agent(self) -> None:
        obs_dim      = self.env.observation_space.shape[0]
        act_dim      = self.env.action_space.shape[0]
        action_limit = float(self.env.action_space.high[0])

        self.agent          = TD3Agent(obs_dim, act_dim, action_limit, self.arch, self.device)
        self.replay         = ReplayBuffer(self.arch.buffer_size)
        self._expl_noise    = self.arch.expl_noise * action_limit
        self._actor_losses:  list[float] = []
        self._critic_losses: list[float] = []

    def _on_episode_start(self) -> None:
        self._actor_losses  = []
        self._critic_losses = []

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        if self.total_steps < self.arch.warmup_steps:
            return self.env.action_space.sample()
        return self.agent.act(obs, noise_std=self._expl_noise)

    def _store_transition(self, obs, action, reward, next_obs, done) -> None:
        self.replay.add(obs, action, reward, next_obs, done)

    def _maybe_update(self) -> None:
        if len(self.replay) >= self.arch.batch_size:
            for _ in range(self.arch.updates_per_step):
                a_loss, c_loss = self.agent.update(self.replay)
                if a_loss != 0.0:
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
