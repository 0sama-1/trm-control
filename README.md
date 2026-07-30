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
salloc -p quantum -N 1 --gres=gpu:2 --time=01:00:00 -A pawsey1302-q
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
_Note: other TRM ablations are available via `model=trm_hier6` / `model=trm_singlez`.
Swap `arch=` for `td3` / `sac` / `ppo`, and `env=` for `cartpole` (continuous-action
algorithms — ddpg/td3/sac/ppo's actor head — currently only support `Box` action
spaces, so `cartpole`'s discrete actions aren't runnable yet)._

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
