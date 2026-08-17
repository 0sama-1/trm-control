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

## Batch Jobs (`train.slurm`, `sweep_models.slurm`, `sweep_archs.slurm`)
Three sbatch scripts at the repo root, all requesting the same resources as
the `salloc` command above and following the same three steps: purge and
reload modules, do a **fresh** `uv` venv install (`bash setup.sh` —
unconditional `rm -rf .venv` + reinstall, every submission), then run via
`torchrun --standalone --nproc_per_node=1`.

- **`train.slurm`** — a single run. Trailing arguments become Hydra
  overrides:
  ```bash
  sbatch train.slurm                                  # defaults: arch=ddpg model=dnn env=pendulum
  sbatch train.slurm arch=td3 model=trm env=cartpole
  sbatch train.slurm arch=ppo model=transformer arch.total_episodes=500
  ```
- **`sweep_models.slurm`** — every model trunk we have (`dnn`, `trm`,
  `trm_hier6`, `trm_singlez`, `transformer`), run sequentially in one job.
  Fixed at `delay=none`, `arch.total_episodes=300`; `arch`/`env` stay at
  their `conf/config.yaml` defaults:
  ```bash
  sbatch sweep_models.slurm
  ```
- **`sweep_archs.slurm`** — every algorithm we have (`ddpg`, `td3`, `sac`,
  `ppo`), run sequentially in one job. `model`/`env`/`delay` stay at their
  defaults, and each arch keeps its own `conf/arch/*.yaml` `total_episodes`
  (200 for ddpg/td3/sac, 2000 for ppo — it's on-policy and needs far more
  per its own config, so this sweep does *not* force a shared count the way
  `sweep_models.slurm` forces 300):
  ```bash
  sbatch sweep_archs.slurm
  ```

Both sweep scripts also forward any trailing arguments, applied to every
run in the sweep (e.g. `sbatch sweep_archs.slurm delay=none` to match
`sweep_models.slurm`'s delay setting).

Since each script does one fresh install per job and then either runs once
(`train.slurm`) or loops sequentially within that same job
(`sweep_*.slurm`), there's no concurrent access to `.venv` to worry about —
unlike a job array, these never have two tasks racing to rebuild it at once.

Logs land in `<job-name>-<job-id>.out`/`.err` in the repo root. `--time` on
the two sweep scripts (`08:00:00` / `16:00:00`) is an untested guess — no
real-hardware timing exists yet for any of these trunks or algorithms (see
`notes.md`) — check the comments at the top of each script for how it was
derived, and expect to adjust it after seeing one real run's wall-clock.
See the wandb prerequisite noted in each script (`src/.env` needs
`WANDB_API_KEY`, and the partition needs outbound internet, unless you pass
`wandb.enabled=false`).

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
