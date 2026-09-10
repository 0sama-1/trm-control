#!/bin/bash
# Phase 2 job submission: BipedalWalker-v3, GPU-accelerated, with ablations
# added per user feedback (not just a bigger environment):
#   - TRM-variant ablation: trm (base), trm_hier6, trm_singlez -- all three
#     already implemented in the codebase, unused until now.
#   - Observation-augmentation on/off ablation at the fixed-delay condition:
#     ties the result back to this project's founding motivation
#     (augmentation.md) rather than leaving a bare "no effect".
#
# Matrix: 6 architectures x 1 environment (BipedalWalker-v3) x 3 conditions
# (no-delay; delay+augmented; delay+unaugmented) x 3 seeds = 54 PBS jobs.
# GPU: ngpus=1 per job (verified working: job 9285343, Tesla V100-32GB).
#
# Run this ON KATANA (from /srv/scratch/z5308482/trm-control).

set -e
cd /srv/scratch/z5308482/trm-control
mkdir -p ~/jobs/phase2 results_phase2 models videos
LOG=~/jobs/phase2_job_ids.txt
: > "$LOG"

ARCHS="dnn transformer lstm trm trm_hier6 trm_singlez"
SEEDS="1 2 3"

submit_job () {
  local model=$1 delay_name=$2 delay_value=$3 augment=$4 seed=$5 tag=$6
  local name="p2_${model}_${tag}_s${seed}"
  local script=~/jobs/phase2/${name}.pbs
  cat > "$script" <<EOF
#!/bin/bash
#PBS -N ${name}
#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1
#PBS -l walltime=11:00:00
#PBS -j oe

module purge; module add python/3.11.3; module add cuda/12.1.1
source /srv/scratch/z5308482/venvs/trm-control/bin/activate
cd /srv/scratch/z5308482/trm-control

python3 main.py \\
  arch=ddpg model=${model} env=bipedalwalker \\
  delay=${delay_name} $( [ -n "$delay_value" ] && echo "delay.value=${delay_value}" ) \\
  augment_obs=${augment} wandb.enabled=false seed=${seed} \\
  print_stat_every=20 record_every=1000 eval_every=20 \\
  results_folder=results_phase2 track_step_history=true
EOF
  local jid
  jid=$(qsub "$script")
  echo "${name} ${jid}" >> "$LOG"
  echo "submitted ${name} -> ${jid}"
}

for model in $ARCHS; do
  for seed in $SEEDS; do
    submit_job "$model" "none"  ""  "true"  "$seed" "none"
    submit_job "$model" "fixed" "1" "true"  "$seed" "fixed-aug"
    submit_job "$model" "fixed" "1" "false" "$seed" "fixed-noaug"
  done
done

echo "Submitted $(wc -l < "$LOG") jobs. IDs recorded in $LOG"
