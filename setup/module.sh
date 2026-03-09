#!bin
module purge
module load pawsey pawseytools pawseyenv/2026.01
module load PrgEnv-nvidia
module load craype-arm-grace
module load gcc-native-mixed/12.3
module load pawseyenv/2026.01
export MODULEPATH=${MODULEPATH}:${LMOD_CUSTOM_COMPILER_GNU_12_0_PREFIX}
module load cuda/12.6
module load python/3.11.6
