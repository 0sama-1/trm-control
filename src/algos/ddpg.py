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
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(r, dtype=np.float32).reshape(-1, 1),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, hidden_dim1: int, hidden_dim2: int):
        super().__init__()
        self.action_limit = action_limit
        self.fc1 = fanin_init(nn.Linear(obs_dim, hidden_dim1))
        self.fc2 = fanin_init(nn.Linear(hidden_dim1, hidden_dim2))
        self.fc3 = nn.Linear(hidden_dim2, act_dim)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_limit


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim1: int, hidden_dim2: int):
        super().__init__()
        self.fc1 = fanin_init(nn.Linear(obs_dim + act_dim, hidden_dim1))
        self.fc2 = fanin_init(nn.Linear(hidden_dim1, hidden_dim2))
        self.fc3 = nn.Linear(hidden_dim2, 1)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, arch_cfg: DictConfig, device: str):
        self.device = device
        self.action_limit = action_limit
        self.cfg = arch_cfg

        self.actor = Actor(obs_dim, act_dim, action_limit, arch_cfg.hidden_dim1, arch_cfg.hidden_dim2).to(device)
        self.critic = Critic(obs_dim, act_dim, arch_cfg.hidden_dim1, arch_cfg.hidden_dim2).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=arch_cfg.actor_lr)
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
            next_a = self.target_actor(ns)
            target_q = self.target_critic(ns, next_a)
            y = r + cfg.gamma * (1.0 - d) * target_q

        critic_loss = nn.functional.mse_loss(self.critic(s, a), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.target_actor, self.actor, cfg.tau)
        soft_update(self.target_critic, self.critic, cfg.tau)

        return actor_loss.item(), critic_loss.item()

    def save(self, path: str, episode_count: int):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "episode_count": episode_count,
            },
            path,
        )

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
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
        name_prefix="ddpg",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    agent = DDPGAgent(obs_dim, act_dim, action_limit, arch, device)
    replay = ReplayBuffer(arch.buffer_size)
    noise = OUNoise(act_dim, arch.ou_noise.theta, arch.ou_noise.sigma, arch.ou_noise.dt)

    episode_start = 0
    if Path(model_path).exists():
        episode_start = agent.load(model_path)
        print(f"Loaded checkpoint from episode {episode_start}")

    total_steps = 0
    recent_returns: list[float] = []

    for ep in range(episode_start, arch.total_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        noise.reset()

        ep_return = 0.0
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for _ in range(env_cfg.max_steps):
            if total_steps < arch.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, noise=noise.sample() * action_limit)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.add(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_return += reward
            total_steps += 1

            if len(replay) >= arch.batch_size:
                for _ in range(arch.updates_per_step):
                    a_loss, c_loss = agent.update(replay)
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
