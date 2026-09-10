# -------------------------
# Module environment (Katana)
# -------------------------
# Pawsey/Setonix-specific modules removed (2026-09-07): this project now
# runs on UNSW Katana (katana.restech.unsw.edu.au), a plain x86_64 PBS Pro
# cluster, not Pawsey's ARM/Cray-EX Setonix. The following no longer apply
# and are NOT loaded here:
#   module load pawsey pawseytools pawseyenv/2026.01   # Pawsey env stack
#   module load PrgEnv-nvidia                          # Cray programming env
#   module load craype-arm-grace                       # ARM Grace target - Katana nodes are x86_64
#   module load gcc-native-mixed/12.3                   # Cray-native compiler wrapper
#   export MODULEPATH=...LMOD_CUSTOM_COMPILER_GNU_12_0_PREFIX  # Cray-specific module path hook
#
# `module purge` itself IS transferable - Katana uses the same
# Environment Modules / Lmod tooling - so it's kept below.

module purge

# Verified working on Katana (2026-09-07, CPU-only job on the mech12 queue):
#   module add python/3.11.3
# `module add` is Katana's alias for `module load`. Module names/versions
# drift over time - always sanity-check with `module avail python` /
# `module avail cuda` on the actual login node before a real run rather
# than trusting the pin below blindly.
module add python/3.11.3

# TODO(verify on Katana): no CUDA module has been confirmed from an actual
# GPU job yet - the katana-stats-job test run was CPU-only (mech12, no
# GPU). Before running GPU training, `ssh katana 'module avail cuda'` and
# pin the result here, e.g.:
#   module add cuda/12.x

echo "Python: $(which python3)"
python3 -V
