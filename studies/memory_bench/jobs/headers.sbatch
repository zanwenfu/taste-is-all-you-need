#!/bin/bash
#SBATCH --job-name=aos-part-headers
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# The model-written commit headers, from answer-free text. No gold and no question is read. Any job can be run again and asks only for what is not stored. Free.
set -uo pipefail
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
JOB=$R/headers/job-$SLURM_JOB_ID; mkdir -p $JOB
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT part $PART"
gpu_log $JOB
serve_oss $JOB 65536 0.92
cd $CODE
for row in "lme $LME_SPLIT $LME_DATA" "locomo $LOC_SPLIT $LOC_DATA"; do
  set -- $row
  T1=$(date +%s)
  $PY -m studies.memory_bench.run headers --benchmark $1 --data $3 --split $2 --part $PART --url http://127.0.0.1:$PORT/v1 --model $OSS_NAME \
    --max-tokens 800 --effort low --workers 48 --out $R/headers/$1.jsonl 2>&1 | tail -1 | cut -c1-240
  echo "$1 headers exit ${PIPESTATUS[0]} seconds $(( $(date +%s) - T1 ))"
done
metrics $JOB
kill $PID $SMI 2>/dev/null; wait $PID 2>/dev/null
echo "end $(date -u +%FT%TZ)"
