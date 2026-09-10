module purge

# TODO(verify before use): interactive-session syntax below is inferred
# from standard PBS Pro conventions and this project's mech/mechgpu queue
# resources (see notes.md), NOT yet run end-to-end on Katana. What IS
# verified from a real job on 2026-09-07 is the batch-submission pattern:
#   #PBS -l select=1:ncpus=1:mem=4gb
#   #PBS -l walltime=00:10:00
#   qsub submit.pbs                 # never run compute on the login node
#   qstat -u $USER                  # poll at ~30-60s intervals, not tight-loop
# Adjust ncpus/ngpus/mem below to what the actual training run needs -
# don't just copy this line.
qsub -I -l select=1:ncpus=1:ngpus=1:mem=32gb -l walltime=01:00:00

---

module purge
module add python/3.11.3

# TODO(verify on Katana): no CUDA module confirmed yet from an actual GPU
# job - run `module avail cuda` on katana and pin it here before GPU training.

project="trm-control"
source /srv/scratch/${USER}/venvs/${project}/bin/activate
source ~/.bashrc

cd /srv/scratch/${USER}/${project}

---

uv venv --clear --system-site-packages /srv/scratch/${USER}/venvs/${project}

---

source /srv/scratch/${USER}/venvs/${project}/bin/activate
uv pip install --upgrade pip

uv pip install -e . \
 --extra-index-url https://download.pytorch.org/whl/cu126 \
 --index-strategy unsafe-best-match

---

# GPU training entry point (single-process, see CLAUDE.md)
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=trm env=pendulum
