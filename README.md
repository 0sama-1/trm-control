# Loading Modules:
Run the following command
```bash
source setup/module.sh
```

# Venv Set Up:
Run the command below to set up the virtual environment (to install the basic pytorch modules)
```bash
export PROJECT="trm-control"
cd $MYSCRATCH/${PROJECT}
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu126
```
Or alternatively, run the following shell command:
```bash
source setup/venv.sh
```
# Useful Sanity-Check Commands
* `uname -m`: gives you the machine hardware architecture of the system.
* `nvidia-smi`: inspects and monitors the Nvidia GPUs on a system
  * `quantum` partition uses GH200 GPUs that use CUDA 12.7 Version. 
  <br>_Note_: If the code does not compile for CUDA incompatibility, make sure the version of the loaded CUDA module $\leq$ version of the GPU.
* `echo $CUDA_VISIBLE_DEVICES`

# Allocating a Computing Node
## Interactive
Using `salloc`:
```bash
salloc -p quantum -N 1 --gres=gpu:1 --time=01:00:00 -A pawsey1302-q
```

## Sending a Command to Scheduler
Using `srun`:
```bash
srun script.sh
```
_Note: Either add the SBATCH directives in the shell file or as flags in the script._

## Running This Repo (`main.py`)
Entry point is `main.py` at the repo root (Hydra config). This script is
single-process — it does not call `torch.distributed.init_process_group` and
`get_device()` in `src/utils.py` always resolves to a single `"cuda"` device,
not a per-rank one — so `--nproc_per_node` must stay at `1`. A higher value
would not parallelise training; it would launch N independent, uncoordinated
copies competing for the same GPU and writing to the same checkpoint file.

- DNN trunk (baseline MLP):
```bash
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=dnn env=pendulum
```
- TRM trunk (recursive-reasoning trunk, see `notes.md`):
```bash
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=trm env=pendulum
```
_Note: other TRM ablations are available via `model=trm_hier6` / `model=trm_singlez`._
- Transformer trunk (multi-head self-attention over per-feature observation tokens, see `notes.md`):
```bash
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=transformer env=pendulum
```
_Note: swap `arch=` for `td3` / `sac` / `ppo`, and `env=` for `cartpole` (continuous-action
algorithms — ddpg/td3/sac/ppo's actor head — currently only support `Box` action
spaces, so `cartpole`'s discrete actions aren't runnable yet)._

## Submitting a Batch Job (`train.slurm`)
For non-interactive runs, `train.slurm` (repo root) wraps the same
`torchrun` invocation above in an sbatch script: it requests the same
resources as the `salloc` command above, loads modules and builds/activates
the venv (via `slurm_env.sh`, shared with `sweep_models.slurm` below), then
runs `main.py`. Any trailing arguments become Hydra overrides:
```bash
sbatch train.slurm                                  # defaults: arch=ddpg model=dnn env=pendulum
sbatch train.slurm arch=td3 model=trm env=cartpole
sbatch train.slurm arch=ppo model=transformer arch.total_episodes=500
```
Logs land in `<job-name>-<job-id>.out`/`.err` in the repo root. See the
wandb prerequisite noted in `slurm_env.sh` (`src/.env` needs
`WANDB_API_KEY`, and the partition needs outbound internet, unless you pass
`wandb.enabled=false`).

## Sweeping All Model Trunks (`sweep_models.slurm`)
`sweep_models.slurm` is a SLURM job array — one array task per model trunk
we have (`dnn`, `trm`, `trm_hier6`, `trm_singlez`, `transformer`), each
getting its own single-GPU allocation so all 5 train in parallel instead of
serially:
```bash
sbatch sweep_models.slurm
```
Fixed at `delay=none`, `arch.total_episodes=300`, `arch`/`env` left at their
`conf/config.yaml` defaults (ddpg / pendulum) — edit the `MODELS=(...)`
array or the final `torchrun` line in the script to change any of that.
`slurm_env.sh`'s venv bootstrap is `flock`-guarded so the 5 array tasks
(which SLURM typically starts around the same time) don't race each other
rebuilding `.venv` — only the first task to grab the lock runs `setup.sh`;
the rest wait, then reuse it. `--time=02:00:00` is an untested guess (1.5x
`train.slurm`'s 1-hour baseline for the 300 vs. 200-episode default) — no
real-hardware timing exists yet for any of these trunks (see `notes.md`),
so adjust once you've seen one run's actual wall-clock.

## Running Program (generic multi-GPU/multi-node reference)
Using `torchrun`:
- Single-Node Multi-GPU Training
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --batch_size 32
```
- Multi-Node Training
```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --rdzv_endpoint="192.168.1.10:29500" train.py
```
