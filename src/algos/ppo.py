import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trainer import OnPolicyTrainer
from utils import atanh, layer_init


# ---------------------------------------------------------------------------
# Model / buffer components
# ---------------------------------------------------------------------------

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
        action_bias  = (action_high + action_low) / 2.0
        action_env   = a_norm * action_scale + action_bias
        log_prob = (dist.log_prob(u) - torch.log(1 - a_norm.pow(2) + 1e-6)).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPOTrainer(OnPolicyTrainer):
    def _build_agent(self) -> None:
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        self.actor  = Actor(obs_dim, act_dim, self.arch.hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, self.arch.hidden_dim).to(self.device)
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=self.arch.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.arch.critic_lr)
        self.buffer = RolloutBuffer(self.arch.rollout_steps, obs_dim, act_dim, self.device)

        self.action_low  = torch.tensor(self.env.action_space.low,  dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)

        self._pg_losses:      list[float] = []
        self._value_losses:   list[float] = []
        self._entropy_values: list[float] = []
        self._last_obs: torch.Tensor | None = None

    def _initial_obs(self) -> torch.Tensor:
        obs, _ = self.env.reset(seed=self.cfg.seed)
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        self.buffer.ptr = 0

        for _ in range(self.arch.rollout_steps):
            with torch.no_grad():
                value = self.critic(obs.unsqueeze(0)).squeeze(0)
                action_env, action_norm, logprob, _ = self.actor.sample(
                    obs.unsqueeze(0), self.action_low, self.action_high
                )

            next_obs_np, reward, terminated, truncated, info = self.env.step(
                action_env.squeeze(0).cpu().numpy()
            )
            done = terminated or truncated
            self.total_steps += 1

            self.buffer.add(
                obs=obs,
                action_norm=action_norm.squeeze(0),
                logprob=logprob.squeeze(0),
                reward=torch.tensor(reward, dtype=torch.float32, device=self.device),
                done=torch.tensor(float(done), dtype=torch.float32, device=self.device),
                value=value,
            )

            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)

            if done:
                self.episode_count += 1
                ep_return = float(info["episode"]["r"])
                avg100 = self._record_return(ep_return)

                if self.episode_count % self.cfg.print_stat_every == 0:
                    print(
                        f"Episode {self.episode_count}/{self.arch.total_episodes} | "
                        f"Return: {ep_return:8.2f} | Avg(100): {avg100:8.2f}"
                    )

                if self.episode_count >= self.arch.total_episodes:
                    break

                next_obs_np, _ = self.env.reset()
                obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)

        self._last_obs = obs
        return obs

    def _update_policy(self) -> None:
        with torch.no_grad():
            last_value = self.critic(self._last_obs.unsqueeze(0)).squeeze(0)
            last_done  = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self.buffer.compute_gae(last_value, last_done, self.arch.gamma, self.arch.gae_lambda)

        b_obs        = self.buffer.obs
        b_actions    = self.buffer.actions_norm
        b_logprobs   = self.buffer.logprobs
        b_advantages = self.buffer.advantages
        b_returns    = self.buffer.returns
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = self.buffer.ptr
        inds = np.arange(batch_size)

        self._pg_losses      = []
        self._value_losses   = []
        self._entropy_values = []

        for _ in range(self.arch.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, self.arch.minibatch_size):
                mb = inds[start : start + self.arch.minibatch_size]

                new_logprobs, entropy = self.actor.evaluate_actions(b_obs[mb], b_actions[mb])
                new_values = self.critic(b_obs[mb])

                ratio = (new_logprobs - b_logprobs[mb]).exp()
                adv   = b_advantages[mb]
                pg_loss    = torch.max(-adv * ratio, -adv * ratio.clamp(1 - self.arch.clip_coef, 1 + self.arch.clip_coef)).mean()
                value_loss = 0.5 * ((new_values - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()

                self.actor_optim.zero_grad()
                (pg_loss - self.arch.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.arch.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                (self.arch.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.arch.max_grad_norm)
                self.critic_optim.step()

                self._pg_losses.append(pg_loss.item())
                self._value_losses.append(value_loss.item())
                self._entropy_values.append(entropy_loss.item())

    def _update_metrics(self) -> dict:
        if not self._pg_losses:
            return {}
        return {
            "pg_loss":    float(np.mean(self._pg_losses)),
            "value_loss": float(np.mean(self._value_losses)),
            "entropy":    float(np.mean(self._entropy_values)),
        }

    def _save_checkpoint(self) -> None:
        torch.save({
            "actor_state_dict":  self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimiser":   self.actor_optim.state_dict(),
            "critic_optimiser":  self.critic_optim.state_dict(),
            "episode_count":     self.episode_count,
        }, self.model_path)

    def _load_checkpoint(self) -> int:
        ckpt = torch.load(self.model_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optim.load_state_dict(ckpt["actor_optimiser"])
        self.critic_optim.load_state_dict(ckpt["critic_optimiser"])
        return ckpt.get("episode_count", 0)
