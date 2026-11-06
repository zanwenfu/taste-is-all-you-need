#!/bin/bash
#SBATCH --job-name=aos-part-judge
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# The free judges, gpt-oss-120b served here at temperature 0: LongMemEval's own answer judge on every answer from both readers, and Mem0's published retrieval judge on what every
# row at the main budget delivered. Any job can be run again. A stage that fails fails the job. Free.
set -uo pipefail
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
JOB=$R/judge-$SLURM_JOB_ID; mkdir -p $JOB
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT part $PART"
gpu_log $JOB
serve_oss $JOB 65536 0.92
cd $CODE
FAILED=0
for A in $R/answers-*/lme/*/answers.jsonl; do
  T1=$(date +%s)
  LOG=$($PY -m studies.memory_bench.run judge --answers $A --reference $LME_REF --url http://127.0.0.1:$PORT/v1 --model $OSS_NAME --max-tokens 512 --effort low --workers 48 2>&1); code=$?
  [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "${A#$R/} exit $code seconds $(( $(date +%s) - T1 )) $(echo "$LOG" | tail -1 | cut -c1-120)"
done
for FOLDER in $R/records/lme/*-8000/; do
  NAME=$(basename $FOLDER); [ "$NAME" = none-8000 ] && continue
  T1=$(date +%s); mkdir -p $R/enough/lme/$NAME
  LOG=$($PY -m studies.memory_bench.run enough --records $FOLDER/records.jsonl --reference $LME_REF --prompts $JUDGE_PROMPTS --url http://127.0.0.1:$PORT/v1 --model $OSS_NAME \
    --workers 48 --out $R/enough/lme/$NAME/verdicts.jsonl 2>&1); code=$?
  [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "enough $NAME exit $code seconds $(( $(date +%s) - T1 )) $(echo "$LOG" | tail -1 | cut -c1-120)"
done
metrics $JOB
kill $PID $SMI 2>/dev/null; wait $PID 2>/dev/null
echo "end $(date -u +%FT%TZ) failed $FAILED"
[ $FAILED = 0 ]
