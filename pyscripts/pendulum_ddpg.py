import os
import copy
import random
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Config
# =========================
@dataclass
class Config:
    env_id: str = "Pendulum-v1"
    total_episodes: int = 200
    max_steps: int = 200

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    buffer_size: int = 200_000
    batch_size: int = 128
    warmup_steps: int = 1000
    updates_per_step: int = 1

    hidden_dim1: int = 256
    hidden_dim2: int = 256

    # OU noise
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    ou_dt: float = 1e-2

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_path: str = "ddpg_pendulum.pt"
    video_folder: str = "videos"
    record_every: int = 25
    stats_print_every: int = 10


cfg = Config()


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Utils
# =========================
def fanin_init(layer: nn.Linear):
    fan_in = layer.weight.data.size(0)
    bound = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)
    return layer


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


class OUNoise:
    def __init__(self, action_dim: int, theta=0.15, sigma=0.2, dt=1e-2):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self):
        self.state.fill(0.0)

    def sample(self):
        x = self.state
        dx = self.theta * (-x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state.astype(np.float32)


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

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


# =========================
# Networks
# =========================
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float):
        super().__init__()
        self.action_limit = action_limit

        self.fc1 = fanin_init(nn.Linear(obs_dim, cfg.hidden_dim1))
        self.fc2 = fanin_init(nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2))
        self.fc3 = nn.Linear(cfg.hidden_dim2, act_dim)

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_limit


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        self.fc1 = fanin_init(nn.Linear(obs_dim + act_dim, cfg.hidden_dim1))
        self.fc2 = fanin_init(nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2))
        self.fc3 = nn.Linear(cfg.hidden_dim2, 1)

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# =========================
# Agent
# =========================
class DDPGAgent:
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, device: str):
        self.device = device
        self.action_limit = action_limit

        self.actor = Actor(obs_dim, act_dim, action_limit).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)

        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def act(self, obs: np.ndarray, noise: np.ndarray | None = None):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]

        if noise is not None:
            action = action + noise

        return np.clip(action, -self.action_limit, self.action_limit)

    def update(self, replay: ReplayBuffer):
        s, a, r, ns, d = replay.sample(cfg.batch_size)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.float32, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        # Critic update
        with torch.no_grad():
            next_a = self.target_actor(ns)
            target_q = self.target_critic(ns, next_a)
            y = r + cfg.gamma * (1.0 - d) * target_q

        current_q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(current_q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        pred_a = self.actor(s)
        actor_loss = -self.critic(s, pred_a).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft target update
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
                "config": cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        return ckpt.get("episode_count", 0)


# =========================
# Main
# =========================
def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.video_folder, exist_ok=True)

    env = gym.make(cfg.env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=cfg.video_folder,
        episode_trigger=lambda ep: (ep + 1) % cfg.record_every == 0,
        name_prefix="ddpg-pendulum",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    agent = DDPGAgent(obs_dim, act_dim, action_limit, cfg.device)
    replay = ReplayBuffer(cfg.buffer_size)
    noise = OUNoise(act_dim, cfg.ou_theta, cfg.ou_sigma, cfg.ou_dt)

    episode_start = 0
    if Path(cfg.model_path).exists():
        episode_start = agent.load(cfg.model_path)
        print(f"Loaded checkpoint from episode {episode_start}")

    total_steps = 0
    recent_returns = []

    for ep in range(episode_start, cfg.total_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        noise.reset()

        ep_return = 0.0
        actor_losses = []
        critic_losses = []

        for step in range(cfg.max_steps):
            if total_steps < cfg.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, noise=noise.sample() * action_limit)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay.add(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_return += reward
            total_steps += 1

            if len(replay) >= cfg.batch_size:
                for _ in range(cfg.updates_per_step):
                    a_loss, c_loss = agent.update(replay)
                    actor_losses.append(a_loss)
                    critic_losses.append(c_loss)

            if done:
                break

        recent_returns.append(ep_return)
        if len(recent_returns) > 100:
            recent_returns.pop(0)

        if (ep + 1) % cfg.stats_print_every == 0:
            avg100 = float(np.mean(recent_returns))
            mean_actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
            mean_critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
            print(
                f"Episode {ep+1:4d}/{cfg.total_episodes} | "
                f"Return: {ep_return:8.2f} | Avg100: {avg100:8.2f} | "
                f"ActorLoss: {mean_actor_loss:9.4f} | CriticLoss: {mean_critic_loss:9.4f}"
            )

        if (ep + 1) % 25 == 0:
            agent.save(cfg.model_path, ep + 1)

    agent.save(cfg.model_path, cfg.total_episodes)
    env.close()

    print("\nTraining complete.")
    print(f"Model saved to: {cfg.model_path}")
    print(f"Videos saved to: {cfg.video_folder}/")


if __name__ == "__main__":
    main()