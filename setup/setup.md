module purge
salloc -p quantum -N 1 --gres=gpu:1 --time=01:00:00 -A pawsey1302-q

salloc -p quantum -N 1 --gres=gpu:2 --time=01:00:00 -A pawsey1302-q

---

module purge
module load pawsey pawseytools pawseyenv/2026.01
module load PrgEnv-nvidia
module load craype-arm-grace
module load gcc-native-mixed/12.3
module load pawseyenv/2026.01
export MODULEPATH=${MODULEPATH}:${LMOD_CUSTOM_COMPILER_GNU_12_0_PREFIX}

module load cuda/12.6
module load python/3.11.6

project="chain-of-strategy"
source $MYSCRATCH/venvs/${project}/bin/activate
source ~/.bashrc

cd $MYSCRATCH/$project

---

uv venv --clear --system-site-packages $MYSCRATCH/venvs/$project

---

source $MYSCRATCH/venvs/${project}/bin/activate
uv pip install --upgrade pip

uv pip install --no-deps $MYSOFTWARE/wheels/flash_attn-2.8.3-cp311-cp311-linux_aarch64.whl

uv pip install -e . \
 --extra-index-url https://download.pytorch.org/whl/cu126 \
 --index-strategy unsafe-best-match

uv pip install torch-c-dlpack-ext

---

accelerate launch $MYSOFTWARE/chain-of-strategy/grpoddp.py
