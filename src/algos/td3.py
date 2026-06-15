import copy
import os
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from utils import fanin_init, get_device, set_seed, soft_update
from delays import build_delay, DelayedEnv


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
        self.actor  = Actor(obs_dim, act_dim, action_limit, h).to(device)
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

        # Target policy smoothing
        with torch.no_grad():
            noise = (torch.randn_like(a) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_a = (self.target_actor(ns) + noise).clamp(-self.action_limit, self.action_limit)
            target_q = torch.min(
                self.target_critic1(ns, next_a),
                self.target_critic2(ns, next_a),
            )
            y = r + cfg.gamma * (1.0 - d) * target_q

        # Critic updates
        c1_loss = nn.functional.mse_loss(self.critic1(s, a), y)
        self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()

        c2_loss = nn.functional.mse_loss(self.critic2(s, a), y)
        self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()

        self._critic_update_count += 1
        actor_loss = 0.0

        # Delayed actor update
        if self._critic_update_count % cfg.policy_delay == 0:
            actor_loss = -self.critic1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            actor_loss = actor_loss.item()

            soft_update(self.target_actor,   self.actor,   cfg.tau)
            soft_update(self.target_critic1, self.critic1, cfg.tau)
            soft_update(self.target_critic2, self.critic2, cfg.tau)

        return actor_loss, c1_loss.item()

    def save(self, path: str, episode_count: int):
        torch.save(
            {
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
            },
            path,
        )

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


def train(cfg: DictConfig):
    arch = cfg.arch
    env_cfg = cfg.env
    device = get_device()

    set_seed(cfg.seed)

    model_path = os.path.join(cfg.model_folder, arch.model_name)
    os.makedirs(cfg.model_folder, exist_ok=True)
    os.makedirs(cfg.video_folder, exist_ok=True)

    env = gym.make(env_cfg.id, render_mode="rgb_array")
    env = DelayedEnv(env, build_delay(cfg.delay))
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=cfg.video_folder,
        episode_trigger=lambda ep: (ep + 1) % cfg.record_every == 0,
        name_prefix="td3",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])
    expl_noise_std = arch.expl_noise * action_limit

    agent = TD3Agent(obs_dim, act_dim, action_limit, arch, device)
    replay = ReplayBuffer(arch.buffer_size)

    episode_start = 0
    if Path(model_path).exists():
        episode_start = agent.load(model_path)
        print(f"Loaded checkpoint from episode {episode_start}")

    total_steps = 0
    recent_returns: list[float] = []

    for ep in range(episode_start, arch.total_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep_return = 0.0
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for _ in range(env_cfg.max_steps):
            if total_steps < arch.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, noise_std=expl_noise_std)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.add(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_return += reward
            total_steps += 1

            if len(replay) >= arch.batch_size:
                for _ in range(arch.updates_per_step):
                    a_loss, c_loss = agent.update(replay)
                    if a_loss != 0.0:
                        actor_losses.append(a_loss)
                    critic_losses.append(c_loss)

            if done:
                break

        recent_returns.append(ep_return)
        if len(recent_returns) > 100:
            recent_returns.pop(0)

        if (ep + 1) % cfg.print_stat_every == 0:
            avg100 = float(np.mean(recent_returns))
            mean_a = float(np.mean(actor_losses)) if actor_losses else 0.0
            mean_c = float(np.mean(critic_losses)) if critic_losses else 0.0
            print(
                f"Episode {ep+1:4d}/{arch.total_episodes} | "
                f"Return: {ep_return:8.2f} | Avg100: {avg100:8.2f} | "
                f"ActorLoss: {mean_a:9.4f} | CriticLoss: {mean_c:9.4f}"
            )

        if (ep + 1) % cfg.record_every == 0:
            agent.save(model_path, ep + 1)

    agent.save(model_path, arch.total_episodes)
    env.close()
    print(f"\nTraining complete. Model: {model_path} | Videos: {cfg.video_folder}/")
