#!/bin/bash
# SLURM batch script — sweeps every algorithm we have, sequentially, in one
# job: ddpg, td3, sac, ppo.
#
# Usage:
#   sbatch sweep_archs.slurm [extra hydra overrides applied to every run]
#
# model/env/delay stay at their conf/config.yaml defaults (dnn / pendulum /
# stochastic), and each arch keeps its own conf/arch/*.yaml total_episodes
# (200 for ddpg/td3/sac, 2000 for ppo — ppo is on-policy and needs far more
# episodes per its own config, so this sweep does NOT force a shared
# episode count the way sweep_models.slurm forces 300). Pass e.g.
# `delay=none` or `arch.total_episodes=300` here if you want this sweep to
# match sweep_models.slurm's settings instead — it applies to every run.
#
# --time is a rough guess, and the shakiest of the three scripts' guesses:
# ddpg/td3/sac at their 200-episode defaults (~1h each, going by
# train.slurm's baseline) plus ppo at its own 2000-episode default (~10x
# that, and on-policy rollouts have a different cost profile than the
# off-policy algos' per-step updates, so this is a guess on top of a
# guess). No real-hardware timing exists yet (see notes.md) — treat this as
# a starting point, and expect to raise it after the first real run.

#SBATCH --job-name=trm-control-sweep-archs
#SBATCH --partition=quantum
#SBATCH --account=pawsey1302-q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# -------------------------
# Purge the node's module environment.
# -------------------------
module purge
module load pawsey pawseytools pawseyenv/2026.01
module load PrgEnv-nvidia
module load craype-arm-grace
module load gcc-native-mixed/12.3
module load pawseyenv/2026.01
export MODULEPATH=${MODULEPATH}:${LMOD_CUSTOM_COMPILER_GNU_12_0_PREFIX}

module load cuda/12.9
module load python/3.11.6

echo "Python: $(which python3)"
python3 -V
nvidia-smi

# -------------------------
# Fresh uv venv, built once for the whole sweep (all 4 runs below reuse it —
# this is one sequential job, so there's no concurrent-job race to guard
# against the way there would be if these ran as separate array tasks).
# -------------------------
bash "$REPO_ROOT/setup.sh"
source "$REPO_ROOT/.venv/bin/activate"
python -V

# wandb is on by default (conf/config.yaml: wandb.enabled=true) and reads
# WANDB_API_KEY from src/.env (see src/logger.py) — make sure that's
# populated, and that this partition has outbound internet, before
# submitting; otherwise pass wandb.enabled=false as one of the overrides.

# -------------------------
# Sweep: every algorithm, each at its own conf/arch/*.yaml defaults
# -------------------------
ARCHS=(ddpg td3 sac ppo)

for ARCH in "${ARCHS[@]}"; do
    echo ""
    echo "=== arch=$ARCH ==="
    echo "Launching: torchrun --standalone --nproc_per_node=1 main.py arch=$ARCH $*"
    torchrun --standalone --nproc_per_node=1 main.py arch="$ARCH" "$@"
done
