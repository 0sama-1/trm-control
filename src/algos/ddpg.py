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

    def add(self, s, a, r, ns, d, *carries):
        self.buffer.append((s, a, r, ns, d, *carries))

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


class CarryReplayBuffer(ReplayBuffer):
    """Replay for trunks that carry a latent across steps (Phase 3 E-2):
    each transition also stores the actor and critic carries the behaviour
    policy held before and after the step, so an update can start every
    trunk from the state it actually acted from (R2D2-style stored state;
    the stored carry goes stale as the weights move, the documented cost)."""

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        cols = list(zip(*batch))
        f = lambda c: np.asarray(np.array(c), dtype=np.float32)
        s, a, r, ns, d = (f(c) for c in cols[:5])
        return (s, a, r.reshape(-1, 1), ns, d.reshape(-1, 1), *(f(c) for c in cols[5:]))


class Actor(nn.Module):
    """Deterministic actor. With a stateful trunk (``trunk.stateful``) the
    forward takes and returns a carry: ``forward(obs, carry) -> (action, carry)``."""

    def __init__(self, obs_dim: int, act_dim: int, action_limit: float, model_cfg: DictConfig):
        super().__init__()
        self.action_limit = action_limit
        self.trunk = instantiate(model_cfg, in_dim=obs_dim)
        self.head  = make_head(self.trunk.out_dim, act_dim, model_cfg.output_init)
        self.stateful = bool(getattr(self.trunk, "stateful", False))

    def initial_carry(self, batch: int, device) -> torch.Tensor:
        return self.trunk.initial_carry(batch, device)

    def forward(self, obs: torch.Tensor, carry: torch.Tensor | None = None):
        if self.stateful:
            h, carry = self.trunk(obs, carry)
            return torch.tanh(self.head(h)) * self.action_limit, carry
        return torch.tanh(self.head(self.trunk(obs))) * self.action_limit


class Critic(nn.Module):
    """Q(s, a). With a stateful trunk: ``forward(obs, act, carry) -> (q, carry)``."""

    def __init__(self, obs_dim: int, act_dim: int, model_cfg: DictConfig):
        super().__init__()
        self.trunk = instantiate(model_cfg, in_dim=obs_dim + act_dim)
        self.head  = make_head(self.trunk.out_dim, 1, model_cfg.output_init)
        self.stateful = bool(getattr(self.trunk, "stateful", False))

    def initial_carry(self, batch: int, device) -> torch.Tensor:
        return self.trunk.initial_carry(batch, device)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, carry: torch.Tensor | None = None):
        x = torch.cat([obs, act], dim=-1)
        if self.stateful:
            h, carry = self.trunk(x, carry)
            return self.head(h), carry
        return self.head(self.trunk(x))

    @property
    def deep_sup(self) -> bool:
        return bool(getattr(self.trunk, "deep_sup", False))

    def forward_cycles(self, obs: torch.Tensor, act: torch.Tensor) -> list[torch.Tensor]:
        """Q readout after every outer recursion cycle (deep supervision, Phase 4 Stage A)."""
        return [self.head(h) for h in self.trunk.forward_cycles(torch.cat([obs, act], dim=-1))]


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

        # Carried latents at acting time (stateful trunks only), batch of 1.
        self.stateful = self.actor.stateful
        self.actor_carry = self.critic_carry = None
        self.reset_carry()

        # Phase 4 Stage A: optional per-network gradient-norm clipping. A value
        # of inf clips nothing but still records the pre-clip norms.
        self.actor_clip = arch_cfg.get("actor_grad_clip")
        self.critic_clip = arch_cfg.get("critic_grad_clip")
        self.grad_norms: dict[str, list[float]] = {"actor": [], "critic": []}
        assert not (self.critic.deep_sup and self.stateful), "deep_sup is stateless-only"

    # -- carried state (stateful trunks) --------------------------------------

    def reset_carry(self) -> None:
        if self.stateful:
            self.actor_carry = self.actor.initial_carry(1, self.device)
            self.critic_carry = self.critic.initial_carry(1, self.device)

    def carries(self) -> tuple[np.ndarray, np.ndarray]:
        return self.actor_carry.cpu().numpy()[0], self.critic_carry.cpu().numpy()[0]

    def advance_critic(self, obs: np.ndarray, action: np.ndarray) -> None:
        """Step the critic's carry on the executed (obs, action), so the critic
        accumulates the same history the actor does."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, self.critic_carry = self.critic(obs_t, act_t, self.critic_carry)

    def act(self, obs: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        """Deterministic action (+ optional noise). Advances the actor carry."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.stateful:
                action, self.actor_carry = self.actor(obs_t, self.actor_carry)
            else:
                action = self.actor(obs_t)
            action = action.cpu().numpy()[0]
        if noise is not None:
            action = action + noise
        return np.clip(action, -self.action_limit, self.action_limit)

    def update(self, replay: ReplayBuffer) -> tuple[float, float]:
        cfg = self.cfg
        batch = replay.sample(cfg.batch_size)
        s, a, r, ns, d = (torch.tensor(x, dtype=torch.float32, device=self.device) for x in batch[:5])

        if self.stateful:
            ca, cc, nca, ncc = (torch.tensor(x, dtype=torch.float32, device=self.device) for x in batch[5:])
            with torch.no_grad():
                next_a, _ = self.target_actor(ns, nca)
                target_q, _ = self.target_critic(ns, next_a, ncc)
                y = r + cfg.gamma * (1.0 - d) * target_q
            critic_loss = nn.functional.mse_loss(self.critic(s, a, cc)[0], y)
            self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()
            actor_loss = -self.critic(s, self.actor(s, ca)[0], cc)[0].mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            soft_update(self.target_actor,  self.actor,  cfg.tau)
            soft_update(self.target_critic, self.critic, cfg.tau)
            return actor_loss.item(), critic_loss.item()

        with torch.no_grad():
            next_a   = self.target_actor(ns)
            target_q = self.target_critic(ns, next_a)
            y = r + cfg.gamma * (1.0 - d) * target_q

        if self.critic.deep_sup:
            # Every outer cycle's readout regresses on the same TD target; the
            # logged loss stays the final cycle's, comparable with other arms.
            qs = self.critic.forward_cycles(s, a)
            losses = [nn.functional.mse_loss(q_k, y) for q_k in qs]
            critic_loss = losses[-1]
            self.critic_opt.zero_grad(); torch.stack(losses).mean().backward()
            self._clip(self.critic, self.critic_clip, "critic"); self.critic_opt.step()
        else:
            critic_loss = nn.functional.mse_loss(self.critic(s, a), y)
            self.critic_opt.zero_grad(); critic_loss.backward()
            self._clip(self.critic, self.critic_clip, "critic"); self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        self._clip(self.actor, self.actor_clip, "actor"); self.actor_opt.step()

        soft_update(self.target_actor,  self.actor,  cfg.tau)
        soft_update(self.target_critic, self.critic, cfg.tau)

        return actor_loss.item(), critic_loss.item()

    def _clip(self, net: nn.Module, max_norm, key: str) -> None:
        if max_norm is None:
            return
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float(max_norm))
        self.grad_norms[key].append(norm.item())

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
        self.replay = (CarryReplayBuffer if self.agent.stateful else ReplayBuffer)(self.arch.buffer_size)
        self._pre_carry: tuple[np.ndarray, np.ndarray] | None = None
        self.noise  = OUNoise(act_dim, self.arch.ou_noise.theta, self.arch.ou_noise.sigma, self.arch.ou_noise.dt)
        self._action_limit = action_limit
        self._actor_losses:  list[float] = []
        self._critic_losses: list[float] = []

    def _on_episode_start(self) -> None:
        self.noise.reset()
        self.agent.reset_carry()
        self._actor_losses  = []
        self._critic_losses = []

    def _on_eval_start(self) -> None:
        # The eval episode runs between training episodes, and the next
        # training episode resets the carry again, so clobbering it is safe.
        self.agent.reset_carry()

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        if self.agent.stateful:
            self._pre_carry = self.agent.carries()
        if self.total_steps < self.arch.warmup_steps:
            action = self.env.action_space.sample()
            if self.agent.stateful:
                self.agent.act(obs)          # still advance the actor's carry
        else:
            action = self.agent.act(obs, noise=self.noise.sample() * self._action_limit)
        if self.agent.stateful:
            self.agent.advance_critic(obs, action)
        return action

    def _select_eval_action(self, obs: np.ndarray) -> np.ndarray:
        return self.agent.act(obs)  # noise=None -> deterministic

    def _store_transition(self, obs, action, reward, next_obs, done) -> None:
        if self.agent.stateful:
            self.replay.add(obs, action, reward, next_obs, done, *self._pre_carry, *self.agent.carries())
        else:
            self.replay.add(obs, action, reward, next_obs, done)

    def _maybe_update(self) -> None:
        if len(self.replay) >= self.arch.batch_size:
            for _ in range(self.arch.updates_per_step):
                a_loss, c_loss = self.agent.update(self.replay)
                self._actor_losses.append(a_loss)
                self._critic_losses.append(c_loss)

    def _episode_metrics(self) -> dict:
        m = {
            "actor_loss":  float(np.mean(self._actor_losses))  if self._actor_losses  else 0.0,
            "critic_loss": float(np.mean(self._critic_losses)) if self._critic_losses else 0.0,
            "buffer_size": len(self.replay),
        }
        # Pre-clip gradient norms of every update this episode (only when
        # clipping is configured), so a clip threshold can be read off a pilot.
        for key, clip in (("actor", self.agent.actor_clip), ("critic", self.agent.critic_clip)):
            if clip is not None:
                norms = self.agent.grad_norms[key]
                m[f"{key}_grad_norms"] = [round(x, 4) for x in norms]
                m[f"{key}_clip_frac"] = float(np.mean([x > float(clip) for x in norms])) if norms else 0.0
                self.agent.grad_norms[key] = []
        return m

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
