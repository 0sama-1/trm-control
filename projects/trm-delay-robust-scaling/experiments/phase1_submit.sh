#!/bin/bash
# Phase 1 job submission: 4 architectures x 2 environments x 2 delay
# conditions x 3 seeds = 48 PBS jobs. Run this ON KATANA (from
# /srv/scratch/z5308482/trm-control) after the smoke test passes.
#
# See projects/trm-delay-robust-scaling/ideas/proposal-final.json for the
# full experiment design this implements.

set -e
cd /srv/scratch/z5308482/trm-control
mkdir -p ~/jobs/phase1 results models videos
LOG=~/jobs/phase1_job_ids.txt
: > "$LOG"

ARCHS="dnn transformer trm lstm"
ENVS="pendulum mountaincar"
SEEDS="1 2 3"

submit_job () {
  local model=$1 env=$2 delay_name=$3 delay_value=$4 seed=$5
  local name="p1_${model}_${env}_${delay_name}_s${seed}"
  local script=~/jobs/phase1/${name}.pbs
  cat > "$script" <<EOF
#!/bin/bash
#PBS -N ${name}
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=06:00:00
#PBS -j oe

module purge; module add python/3.11.3
source /srv/scratch/z5308482/venvs/trm-control/bin/activate
cd /srv/scratch/z5308482/trm-control

python3 main.py \\
  arch=ddpg model=${model} env=${env} \\
  delay=${delay_name} $( [ -n "$delay_value" ] && echo "delay.value=${delay_value}" ) \\
  augment_obs=true wandb.enabled=false seed=${seed} \\
  print_stat_every=20 record_every=1000 eval_every=20
EOF
  local jid
  jid=$(qsub "$script")
  echo "${name} ${jid}" >> "$LOG"
  echo "submitted ${name} -> ${jid}"
}

for model in $ARCHS; do
  for env in $ENVS; do
    for seed in $SEEDS; do
      submit_job "$model" "$env" "none" "" "$seed"
      submit_job "$model" "$env" "fixed" "1" "$seed"
    done
  done
done

echo "Submitted $(wc -l < "$LOG") jobs. IDs recorded in $LOG"
