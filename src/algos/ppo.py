import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from utils import atanh, get_device, layer_init, set_seed
from delays import build_delay, DelayedEnv


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

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mu = self.mu_head(x)
        log_std = self.log_std.expand_as(mu).clamp(-20, 2)
        return mu, torch.exp(log_std)

    def sample(self, obs: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a_norm = torch.tanh(u)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        action_env = a_norm * action_scale + action_bias

        log_prob = (dist.log_prob(u) - torch.log(1 - a_norm.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action_env, a_norm, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, action_normalized: torch.Tensor):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        u = atanh(action_normalized)
        log_prob = (dist.log_prob(u) - torch.log(1 - action_normalized.pow(2) + 1e-6)).sum(dim=-1)
        return log_prob, dist.entropy().sum(dim=-1)


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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int, act_dim: int, device: str):
        self.rollout_steps = rollout_steps
        self.device = device
        self.obs          = torch.zeros((rollout_steps, obs_dim), dtype=torch.float32, device=device)
        self.actions_norm = torch.zeros((rollout_steps, act_dim), dtype=torch.float32, device=device)
        self.logprobs     = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.rewards      = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.dones        = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.values       = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.advantages   = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.returns      = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, action_norm, logprob, reward, done, value):
        self.obs[self.ptr]          = obs
        self.actions_norm[self.ptr] = action_norm
        self.logprobs[self.ptr]     = logprob
        self.rewards[self.ptr]      = reward
        self.dones[self.ptr]        = done
        self.values[self.ptr]       = value
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor, last_done: torch.Tensor, gamma: float, gae_lambda: float):
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
        name_prefix="ppo",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low  = torch.tensor(env.action_space.low,  dtype=torch.float32, device=device)
    action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

    actor  = Actor(obs_dim, act_dim, arch.hidden_dim).to(device)
    critic = Critic(obs_dim, arch.hidden_dim).to(device)
    actor_optim  = optim.Adam(actor.parameters(),  lr=arch.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=arch.critic_lr)

    buffer = RolloutBuffer(arch.rollout_steps, obs_dim, act_dim, device)

    episode_count = 0
    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        actor_optim.load_state_dict(ckpt["actor_optimiser"])
        critic_optim.load_state_dict(ckpt["critic_optimiser"])
        episode_count = ckpt.get("episode_count", 0)
        print(f"Loaded checkpoint from episode {episode_count}")

    obs, _ = env.reset(seed=cfg.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    recent_returns: list[float] = []

    while episode_count < arch.total_episodes:
        buffer.ptr = 0

        for _ in range(arch.rollout_steps):
            with torch.no_grad():
                value = critic(obs.unsqueeze(0)).squeeze(0)
                action_env, action_norm, logprob, _ = actor.sample(obs.unsqueeze(0), action_low, action_high)

            next_obs, reward, terminated, truncated, info = env.step(action_env.squeeze(0).cpu().numpy())
            done = terminated or truncated

            buffer.add(
                obs=obs,
                action_norm=action_norm.squeeze(0),
                logprob=logprob.squeeze(0),
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=torch.tensor(float(done), dtype=torch.float32, device=device),
                value=value,
            )

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

            if done:
                episode_count += 1
                ep_return = float(info["episode"]["r"])
                recent_returns.append(ep_return)
                if len(recent_returns) > 100:
                    recent_returns.pop(0)

                if episode_count % cfg.print_stat_every == 0:
                    avg100 = np.mean(recent_returns) if recent_returns else 0.0
                    print(
                        f"Episode {episode_count}/{arch.total_episodes} | "
                        f"Last Return: {ep_return:8.2f} | Avg(100): {avg100:8.2f}"
                    )

                if episode_count >= arch.total_episodes:
                    break

                next_obs, _ = env.reset()
                obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            last_value = critic(obs.unsqueeze(0)).squeeze(0)
            last_done = torch.tensor(0.0, dtype=torch.float32, device=device)

        buffer.compute_gae(last_value, last_done, arch.gamma, arch.gae_lambda)

        b_obs, b_actions_norm, b_logprobs, b_advantages, b_returns = (
            buffer.obs, buffer.actions_norm, buffer.logprobs, buffer.advantages, buffer.returns
        )
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = buffer.ptr
        inds = np.arange(batch_size)

        for _ in range(arch.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, arch.minibatch_size):
                mb = inds[start : start + arch.minibatch_size]

                new_logprobs, entropy = actor.evaluate_actions(b_obs[mb], b_actions_norm[mb])
                new_values = critic(b_obs[mb])

                ratio = (new_logprobs - b_logprobs[mb]).exp()
                adv = b_advantages[mb]
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1.0 - arch.clip_coef, 1.0 + arch.clip_coef),
                ).mean()

                value_loss = 0.5 * ((new_values - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()

                actor_optim.zero_grad()
                (pg_loss - arch.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), arch.max_grad_norm)
                actor_optim.step()

                critic_optim.zero_grad()
                (arch.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), arch.max_grad_norm)
                critic_optim.step()

    torch.save(
        {
            "actor_state_dict":  actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "actor_optimiser":   actor_optim.state_dict(),
            "critic_optimiser":  critic_optim.state_dict(),
            "episode_count":     episode_count,
        },
        model_path,
    )
    env.close()
    print(f"\nTraining complete. Model: {model_path} | Videos: {cfg.video_folder}/")
