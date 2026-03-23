import os
import math
import random
from dataclasses import dataclass
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
    total_episodes: int = 2000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    ppo_epochs: int = 10
    minibatch_size: int = 64
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    hidden_dim: int = 128
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    video_folder: str = "videos"
    model_path: str = "ppo_pendulum.pt"
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
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * torch.log((1 + x) / (1 - x))


# =========================
# Networks
# =========================
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mu_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mu = self.mu_head(x)
        log_std = self.log_std.expand_as(mu).clamp(-20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor):
        """
        Returns:
            action_env: scaled to env bounds
            action_normalized: in [-1, 1]
            log_prob
            entropy
        """
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)

        # Reparameterization
        u = dist.rsample()
        a_norm = torch.tanh(u)  # [-1, 1]

        # Scale to env bounds
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        action_env = a_norm * action_scale + action_bias

        # Log prob correction for tanh squashing
        # log pi(a) = log pi(u) - sum(log(1 - tanh(u)^2))
        log_prob = dist.log_prob(u) - torch.log(1 - a_norm.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action_env, a_norm, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, action_normalized: torch.Tensor):
        """
        Recompute log probs for PPO updates using stored normalized action.
        """
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)

        u = atanh(action_normalized)
        log_prob = dist.log_prob(u) - torch.log(1 - action_normalized.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs).squeeze(-1)


# =========================
# Rollout Buffer
# =========================
class RolloutBuffer:
    def __init__(self, rollout_steps, obs_dim, act_dim, device):
        self.rollout_steps = rollout_steps
        self.device = device

        self.obs = torch.zeros((rollout_steps, obs_dim), dtype=torch.float32, device=device)
        self.actions_norm = torch.zeros((rollout_steps, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.values = torch.zeros(rollout_steps, dtype=torch.float32, device=device)

        self.advantages = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.returns = torch.zeros(rollout_steps, dtype=torch.float32, device=device)

        self.ptr = 0

    def add(self, obs, action_norm, logprob, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions_norm[self.ptr] = action_norm
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, last_done, gamma, gae_lambda):
        last_gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get(self):
        return (
            self.obs,
            self.actions_norm,
            self.logprobs,
            self.advantages,
            self.returns,
            self.values,
        )


# =========================
# Main Training
# =========================
def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.video_folder, exist_ok=True)

    env = gym.make(cfg.env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=cfg.video_folder,
        episode_trigger=lambda ep: (ep + 1) % 100 == 0,
        name_prefix="ppo-pendulum",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=cfg.device)
    action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=cfg.device)

    actor = Actor(obs_dim, act_dim, cfg.hidden_dim).to(cfg.device)
    critic = Critic(obs_dim, cfg.hidden_dim).to(cfg.device)

    actor_optim = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    buffer = RolloutBuffer(cfg.rollout_steps, obs_dim, act_dim, cfg.device)

    obs, _ = env.reset(seed=cfg.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=cfg.device)

    episode_count = 0
    
    # importing a trained model if path exists
    model_path = Path(cfg.model_path)
    if model_path.exists():
      checkpoint = torch.load("ppo_pendulum.pt", map_location=cfg.device)

      actor.load_state_dict(checkpoint["actor_state_dict"])
      critic.load_state_dict(checkpoint["critic_state_dict"])

      actor_optim.load_state_dict(checkpoint["actor_optimiser"])
      critic_optim.load_state_dict(checkpoint["critic_optimiser"])

      episode_count = checkpoint.get("episode_count", 0)

    recent_returns = []

    while episode_count < cfg.total_episodes:
        buffer.ptr = 0

        for step in range(cfg.rollout_steps):
            with torch.no_grad():
                value = critic(obs.unsqueeze(0)).squeeze(0)
                action_env, action_norm, logprob, _ = actor.sample(
                    obs.unsqueeze(0), action_low, action_high
                )

            action_np = action_env.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(
                obs=obs,
                action_norm=action_norm.squeeze(0),
                logprob=logprob.squeeze(0),
                reward=torch.tensor(reward, dtype=torch.float32, device=cfg.device),
                done=torch.tensor(float(done), dtype=torch.float32, device=cfg.device),
                value=value,
            )

            obs = torch.tensor(next_obs, dtype=torch.float32, device=cfg.device)

            if done:
                episode_count += 1
                ep_return = info["episode"]["r"]
                recent_returns.append(float(ep_return))
                if len(recent_returns) > 100:
                    recent_returns.pop(0)

                if episode_count % cfg.stats_print_every == 0:
                    avg100 = np.mean(recent_returns) if recent_returns else 0.0
                    print(
                        f"Episode {episode_count}/{cfg.total_episodes} | "
                        f"Last Return: {float(ep_return):8.2f} | "
                        f"Avg(100): {avg100:8.2f}"
                    )

                if episode_count >= cfg.total_episodes:
                    break

                next_obs, _ = env.reset()
                obs = torch.tensor(next_obs, dtype=torch.float32, device=cfg.device)

        with torch.no_grad():
            last_value = critic(obs.unsqueeze(0)).squeeze(0)
            last_done = torch.tensor(0.0, dtype=torch.float32, device=cfg.device)

        buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=last_done,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        b_obs, b_actions_norm, b_logprobs, b_advantages, b_returns, b_values = buffer.get()

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = buffer.ptr
        inds = np.arange(batch_size)

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions_norm = b_actions_norm[mb_inds]
                mb_old_logprobs = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]

                new_logprobs, entropy = actor.evaluate_actions(mb_obs, mb_actions_norm)
                new_values = critic(mb_obs)

                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                total_actor_loss = actor_loss - cfg.ent_coef * entropy_loss
                total_critic_loss = cfg.vf_coef * value_loss

                actor_optim.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                actor_optim.step()

                critic_optim.zero_grad()
                total_critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                critic_optim.step()

    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "actor_optimiser": actor_optim.state_dict(),
            "critic_optimiser": critic_optim.state_dict(),
            "config": cfg.__dict__,
            "episode_count": episode_count
        },
        cfg.model_path,
    )

    env.close()
    print(f"\nTraining complete.")
    print(f"Model saved to: {cfg.model_path}")
    print(f"Videos saved to: {cfg.video_folder}/")


if __name__ == "__main__":
    main()