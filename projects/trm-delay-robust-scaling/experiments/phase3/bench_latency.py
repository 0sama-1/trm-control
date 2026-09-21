"""Phase 3 Stage 0 -- per-decision inference latency (plan §3.3 / §4.1, item E-6).

Deployment cost is the actor's forward pass per control decision, not training
wall-clock. For every trunk this times, at Pendulum-v1's input sizes:

  cpu1_b1    CPU, 1 thread, batch 1        -- the embedded-controller case
  cpu1_b256  CPU, 1 thread, batch 256
  gpu_b1     GPU, batch 1, host->device->host round trip included
  gpu_b256   GPU, batch 256, same
  update     one full DDPG update (critic + actor backward, batch 128) per device

and converts p99 decision latency into the highest control rate it admits.
Weights are untrained: latency depends on the architecture, not the values.

  OMP_NUM_THREADS=1 python bench_latency.py --out latency.json
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    while not (p / "main.py").exists():
        p = p.parent
    return p


REPO = _repo_root()
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from algos.ddpg import CarryReplayBuffer, DDPGAgent, ReplayBuffer  # noqa: E402

MODELS = ["dnn", "dnn_large", "trm_small", "trm", "trm_h1", "trm_h6",
          "trm_hier6", "trm_singlez", "transformer", "lstm", "trm_carryz", "lstm_carry"]
# (label, obs_dim, act_dim): Pendulum with fixed d=1 augmentation, and with
# U{0..4} stochastic-delay augmentation.
SHAPES = [("pendulum_fixed1_aug", 4, 1), ("pendulum_stoch4_aug", 7, 1)]
RATES_HZ = [100, 500, 1000]


def _time_calls(fn, n_warm: int, n: int, sync) -> np.ndarray:
    for _ in range(n_warm):
        fn()
    sync()
    out = np.empty(n)
    for i in range(n):
        t = time.perf_counter_ns()
        fn()
        sync()
        out[i] = (time.perf_counter_ns() - t) / 1e6
    return out


def _stats(ms: np.ndarray) -> dict:
    p99 = float(np.percentile(ms, 99))
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p99_ms": p99,
        "mean_ms": float(ms.mean()),
        "max_rate_hz_at_p99": 1000.0 / p99,
        "admits_hz": {str(r): bool(p99 < 1000.0 / r) for r in RATES_HZ},
    }


def bench(model: str, obs_dim: int, act_dim: int, device: str, n: int) -> dict:
    mc = OmegaConf.load(REPO / "conf" / "model" / f"{model}.yaml")
    arch = OmegaConf.load(REPO / "conf" / "arch" / "ddpg.yaml")
    agent = DDPGAgent(obs_dim, act_dim, 2.0, arch, mc, device)
    actor = agent.actor.eval()
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    res = {}
    for b in (1, 256):
        host = np.random.randn(b, obs_dim).astype(np.float32)
        carry = actor.initial_carry(b, device) if actor.stateful else None

        @torch.no_grad()
        def decide():
            # A real decision starts and ends on the host: include the copies.
            x = torch.from_numpy(host).to(device)
            return (actor(x, carry)[0] if actor.stateful else actor(x)).cpu()

        res[f"b{b}"] = _stats(_time_calls(decide, 50, n if b == 1 else max(n // 4, 50), sync))

    buf = (CarryReplayBuffer if agent.stateful else ReplayBuffer)(1024)
    carries = [np.zeros(agent.actor.trunk.carry_dim), np.zeros(agent.critic.trunk.carry_dim)] * 2 if agent.stateful else []
    for _ in range(1024):
        buf.add(np.random.randn(obs_dim), np.random.randn(act_dim), 0.0, np.random.randn(obs_dim), 0.0, *carries)
    agent.actor.train()
    upd = _time_calls(lambda: agent.update(buf), 10, max(n // 10, 30), sync)
    res["update_b128"] = {"p50_ms": float(np.percentile(upd, 50)), "mean_ms": float(upd.mean())}
    res["actor_params"] = sum(p.numel() for p in agent.actor.parameters())
    res["actor_critic_params"] = res["actor_params"] + sum(p.numel() for p in agent.critic.parameters())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_p3s0_latency/latency.json")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--n", type=int, default=2000, help="timed batch-1 calls per cell")
    args = ap.parse_args()

    out = {
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": {},
    }
    try:
        with open("/proc/cpuinfo") as f:
            out["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for label, o, a in SHAPES:
        for model in args.models:
            for dev in devices:
                if dev == "cpu":
                    torch.set_num_threads(1)
                t0 = time.time()
                r = bench(model, o, a, dev, args.n)
                key = f"{label}/{model}/{'cpu1' if dev == 'cpu' else 'gpu'}"
                out["results"][key] = r
                print(f"{key}: b1 p50 {r['b1']['p50_ms']:.3f} ms p99 {r['b1']['p99_ms']:.3f} ms | "
                      f"b256 p50 {r['b256']['p50_ms']:.3f} ms | update {r['update_b128']['p50_ms']:.2f} ms "
                      f"| {time.time() - t0:.0f}s", flush=True)

    dest = Path(args.out)
    dest = dest if dest.is_absolute() else REPO / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
