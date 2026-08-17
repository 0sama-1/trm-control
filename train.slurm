#!/bin/bash
# SLURM batch script — single training run via torchrun on a Pawsey GH200 node.
#
# Usage:
#   sbatch train.slurm [hydra overrides...]
#
# Examples:
#   sbatch train.slurm                                     # defaults: arch=ddpg model=dnn env=pendulum
#   sbatch train.slurm arch=td3 model=trm env=cartpole
#   sbatch train.slurm arch=ppo model=transformer arch.total_episodes=500
#
# For sweeps, see sweep_models.slurm (all model trunks) and
# sweep_archs.slurm (all algorithms).
#
# Resource request mirrors the interactive salloc command used to develop
# this repo (see CLAUDE.md):
#   salloc -p quantum -N 1 --gres=gpu:1 --time=01:00:00 -A pawsey1302-q
# main.py is single-process (see CLAUDE.md's note on torchrun --nproc_per_node),
# so this only ever requests one GPU — bump #SBATCH --time for a real training
# run, 1 hour is just the interactive-session default it's mirroring.

#SBATCH --job-name=trm-control
#SBATCH --partition=quantum
#SBATCH --account=pawsey1302-q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# -------------------------
# Purge the node's module environment — a SLURM batch job starts from a
# clean shell, but purge explicitly anyway so this doesn't depend on
# whatever was (or wasn't) loaded in the submitting shell.
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
# Fresh uv venv — setup.sh unconditionally rm -rf's and rebuilds .venv, so
# calling it here always starts this run from a clean install rather than
# reusing whatever a previous job left behind.
# -------------------------
bash "$REPO_ROOT/setup.sh"
source "$REPO_ROOT/.venv/bin/activate"
python -V

# wandb is on by default (conf/config.yaml: wandb.enabled=true) and reads
# WANDB_API_KEY from src/.env (see src/logger.py) — make sure that's
# populated, and that this partition has outbound internet, before
# submitting; otherwise pass wandb.enabled=false as one of the overrides.

# -------------------------
# Training
# -------------------------
echo "Launching: torchrun --standalone --nproc_per_node=1 main.py $*"
torchrun --standalone --nproc_per_node=1 main.py "$@"
