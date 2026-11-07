#!/bin/bash
#SBATCH --job-name=aos-gpus
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
# The study's GPUs through the days the queue is full, since a two-GPU job can wait for days and the real test has its hour: what runs on them are steps, a server (serve.sbatch) and the work
# that reads from it, so that one reader can follow another on the same GPUs. Never kept idle: it is ended by hand, by its job id, when no work is due. Free, no key.
set -uo pipefail
D=${STUDY_ROOT:?}; OUT=$D/logs/gpus-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date -u +%FT%TZ) on $(hostname)"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 60 > $OUT/gpu.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null; exit 143' TERM
sleep 8640000 &
wait $!
