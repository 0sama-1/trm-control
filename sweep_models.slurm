#!/bin/bash
# SLURM batch script — sweeps every model trunk we have, sequentially, in
# one job: dnn, trm, trm_hier6, trm_singlez, transformer. no delay, 300
# episodes each.
#
# Usage:
#   sbatch sweep_models.slurm [extra hydra overrides applied to every run]
#
# Fixed per run: delay=none, arch.total_episodes=300. arch/env stay at their
# conf/config.yaml defaults (ddpg / pendulum) — this only sweeps the model
# trunk. Edit MODELS= below to change the set, or pass e.g. `env=cartpole`
# to apply it to every run in the sweep.
#
# --time is a rough guess: ~1.5x train.slurm's 1-hour baseline per model
# (300 vs. the usual 200-episode default) x 5 models run back to back in
# this one job, plus setup overhead, rounded up. No real-hardware timing
# exists yet for any of these trunks (see notes.md) — treat this as a
# starting point and adjust once you've seen actual wall-clock.

#SBATCH --job-name=trm-control-sweep-models
#SBATCH --partition=quantum
#SBATCH --account=pawsey1302-q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# -------------------------
# Purge the node's module environment.
# -------------------------
source setup/.module

# -------------------------
# Fresh uv venv, built once for the whole sweep (all 5 runs below reuse it —
# this is one sequential job, so there's no concurrent-job race to guard
# against the way there would be if these ran as separate array tasks).
# -------------------------
source setup/.uv-venv
source .venv/bin.activate

# wandb is on by default (conf/config.yaml: wandb.enabled=true) and reads
# WANDB_API_KEY from src/.env (see src/logger.py) — make sure that's
# populated, and that this partition has outbound internet, before
# submitting; otherwise pass wandb.enabled=false as one of the overrides.

# -------------------------
# Sweep: every model trunk, no delay, 300 episodes each
# -------------------------
MODELS=(dnn trm trm_hier6 trm_singlez transformer)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=== model=$MODEL ==="
    echo "Launching: torchrun --standalone --nproc_per_node=1 main.py model=$MODEL delay=none arch.total_episodes=300 $*"
    torchrun --standalone --nproc_per_node=1 main.py model="$MODEL" delay=none arch.total_episodes=300 "$@"
done
