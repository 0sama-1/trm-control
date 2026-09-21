import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from trainer import OffPolicyTrainer
from utils import soft_update


# ---------------------------------------------------------------------------
# Model / agent components
# ---------------------------------------------------------------------------

# Identical to the definitions this file used to carry; imported so TD3 shares
# DDPG's stateful-trunk (Carry-Z) support, see algos/ddpg.py.
from algos.ddpg import Actor, CarryReplayBuffer, Critic, ReplayBuffer  # noqa: E402


class TD3Agent:
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, arch_cfg: DictConfig, model_cfg: DictConfig, device: str):
        self.device = device
        self.action_limit = action_limit
        self.cfg = arch_cfg
        self._critic_update_count = 0

        self.actor   = Actor(obs_dim, act_dim, action_limit, model_cfg).to(device)
        self.critic1 = Critic(obs_dim, act_dim, model_cfg).to(device)
        self.critic2 = Critic(obs_dim, act_dim, model_cfg).to(device)

        self.target_actor   = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=arch_cfg.actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=arch_cfg.critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=arch_cfg.critic_lr)

        # Carried latents at acting time (stateful trunks only): actor + both critics.
        self.stateful = self.actor.stateful
        self._carry = None
        self.reset_carry()

    # -- carried state (stateful trunks), same scheme as DDPGAgent ------------

    def reset_carry(self) -> None:
        if self.stateful:
            self._carry = [m.initial_carry(1, self.device) for m in (self.actor, self.critic1, self.critic2)]

    def carries(self) -> tuple[np.ndarray, ...]:
        return tuple(c.cpu().numpy()[0] for c in self._carry)

    def advance_critic(self, obs: np.ndarray, action: np.ndarray) -> None:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, self._carry[1] = self.critic1(obs_t, act_t, self._carry[1])
            _, self._carry[2] = self.critic2(obs_t, act_t, self._carry[2])

    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.stateful:
                action, self._carry[0] = self.actor(obs_t, self._carry[0])
            else:
                action = self.actor(obs_t)
            action = action.cpu().numpy()[0]
        if noise_std > 0.0:
            action += np.random.randn(*action.shape).astype(np.float32) * noise_std
        return np.clip(action, -self.action_limit, self.action_limit)

    def update(self, replay: ReplayBuffer) -> tuple[float, float]:
        cfg = self.cfg
        batch = replay.sample(cfg.batch_size)
        s, a, r, ns, d = (torch.tensor(x, dtype=torch.float32, device=self.device) for x in batch[:5])
        if self.stateful:
            ca, c1, c2, nca, nc1, nc2 = (torch.tensor(x, dtype=torch.float32, device=self.device) for x in batch[5:])
            q = lambda net, o, act, c: net(o, act, c)[0]
            pi = lambda net, o, c: net(o, c)[0]
        else:
            ca = c1 = c2 = nca = nc1 = nc2 = None
            q = lambda net, o, act, c: net(o, act)
            pi = lambda net, o, c: net(o)

        with torch.no_grad():
            noise  = (torch.randn_like(a) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_a = (pi(self.target_actor, ns, nca) + noise).clamp(-self.action_limit, self.action_limit)
            target_q = torch.min(q(self.target_critic1, ns, next_a, nc1), q(self.target_critic2, ns, next_a, nc2))
            y = r + cfg.gamma * (1.0 - d) * target_q

        c1_loss = nn.functional.mse_loss(q(self.critic1, s, a, c1), y)
        self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()

        c2_loss = nn.functional.mse_loss(q(self.critic2, s, a, c2), y)
        self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()

        self._critic_update_count += 1
        actor_loss = 0.0

        if self._critic_update_count % cfg.policy_delay == 0:
            actor_loss = -q(self.critic1, s, pi(self.actor, s, ca), c1).mean()
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

        self.agent          = TD3Agent(obs_dim, act_dim, action_limit, self.arch, self.cfg.model, self.device)
        self.replay         = (CarryReplayBuffer if self.agent.stateful else ReplayBuffer)(self.arch.buffer_size)
        self._pre_carry     = None
        self._expl_noise    = self.arch.expl_noise * action_limit
        self._actor_losses:  list[float] = []
        self._critic_losses: list[float] = []

    def _on_episode_start(self) -> None:
        self.agent.reset_carry()
        self._actor_losses  = []
        self._critic_losses = []

    def _on_eval_start(self) -> None:
        self.agent.reset_carry()

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        if self.agent.stateful:
            self._pre_carry = self.agent.carries()
        if self.total_steps < self.arch.warmup_steps:
            action = self.env.action_space.sample()
            if self.agent.stateful:
                self.agent.act(obs)          # still advance the actor's carry
        else:
            action = self.agent.act(obs, noise_std=self._expl_noise)
        if self.agent.stateful:
            self.agent.advance_critic(obs, action)
        return action

    def _select_eval_action(self, obs: np.ndarray) -> np.ndarray:
        return self.agent.act(obs)  # noise_std=0.0 -> deterministic

    def _store_transition(self, obs, action, reward, next_obs, done) -> None:
        if self.agent.stateful:
            self.replay.add(obs, action, reward, next_obs, done, *self._pre_carry, *self.agent.carries())
        else:
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
