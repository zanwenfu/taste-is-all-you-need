#!/bin/bash
#SBATCH --job-name=aos-part-swe-read
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# What swe.sbatch left of the coding set: each answer and each verdict with none on record, at the settings of the first run. Any job can be run again and asks only what has no answer.
# A row that fails fails the job. Then the table, which refuses a run that is not whole. Free.
set -uo pipefail
export PART=swe
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
C=$R/cases/cases.jsonl; JOB=$R/read-$SLURM_JOB_ID; mkdir -p $JOB
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT"
gpu_log $JOB
serve_oss $JOB 131072 0.92
cd $CODE
FAILED=0
for BLOCKS in $R/blocks/*.jsonl; do
  NAME=$(basename $BLOCKS .jsonl); T1=$(date +%s)
  LOG=$($PY -m studies.memory_bench.swe answer --blocks $BLOCKS --cases $C --url http://127.0.0.1:$PORT/v1 --model $OSS_NAME --workers 24 --out $R/answers/$NAME.jsonl 2>&1); code=$?
  [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "$NAME exit $code seconds $(( $(date +%s) - T1 )) $(echo "$LOG" | tail -1 | cut -c1-160)"
  [ -d $R/judged/$NAME ] || continue
  LOG=$($INSIDE $D/venv/bin/python -m studies.memory_probe.judge --results $R/judged/$NAME/results.jsonl --cases $C --url http://127.0.0.1:$PORT/v1 --model $OSS_NAME --workers 24 2>&1); code=$?
  [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "judge $NAME exit $code $(echo "$LOG" | tail -1 | cut -c1-160)"
done
metrics $JOB
kill $PID $SMI 2>/dev/null; wait $PID 2>/dev/null
$PY -m studies.memory_bench.swe report --root $R --commit $SHA --digest $DIGEST --out $R/report.md || FAILED=$((FAILED + 1))
echo "end $(date -u +%FT%TZ) failed $FAILED"
[ $FAILED = 0 ]
