#!/bin/bash
#SBATCH --job-name=aos-part-read
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
# One reader answers every record meant for it, one row at a time, at the output cap the analysis holds it to. READER is oss, or llama on two GPUs. Any job can be run again and
# asks only what has no answer. ONLY and NOT limit the rows by a grep pattern, so several jobs can share the work. Then LoCoMo's judge-free score. A row whose records did not
# pass their checks has no summary and is not read. A row that fails fails the job. Free.
set -uo pipefail
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
READER=${READER:?oss or llama}; ONLY=${ONLY:-.}; NOT=${NOT:-^$}; WORKERS=${WORKERS:-24}
OUT=$R/answers-$READER; JOB=$OUT/job-$SLURM_JOB_ID; mkdir -p $JOB
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT part $PART reader $READER rows '$ONLY' not '$NOT'"
gpu_log $JOB
if [ $READER = llama ]; then
  MODEL=$LLAMA_NAME; PORT=$((20000 + RANDOM % 20000)); t0=$(date +%s)
  $RUN_GPU vllm serve $LLAMA --served-model-name $MODEL --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.92 --host 127.0.0.1 --port $PORT > $JOB/vllm.log 2>&1 &
  PID=$!
  wait_for $JOB $PORT $t0
else
  MODEL=$OSS_NAME
  serve_oss $JOB 131072 0.92
fi
cd $CODE
LIST=$($PY - $R $READER <<'PY'
import sys
from pathlib import Path
from studies.memory_bench.analyze import reader_cap, readers_of
root, reader = Path(sys.argv[1]), sys.argv[2]
for view in ("lme", "locomo"):
    for folder in sorted((root / "records" / view).iterdir()):
        if folder.is_dir() and reader in readers_of(folder.name) and (folder / "records.summary.json").exists():
            cap, effort = reader_cap(reader, view, folder.name)
            print(f"{view}/{folder.name} {cap} {effort or ''}")
PY
)
FAILED=0
while read -r ENTRY MAX EFFORT; do
  [ -n "$ENTRY" ] || continue
  VIEW=${ENTRY%%/*}; NAME=${ENTRY##*/}; W=$WORKERS; T1=$(date +%s); EXTRA=(); ans=$OUT/$VIEW/$NAME; mkdir -p $ans
  case $NAME in full_llama*) W=3;; full_oss*) W=4;; oracle_*) W=12;; esac
  [ -n "$EFFORT" ] && EXTRA=(--effort $EFFORT)
  $PY -m studies.memory_bench.run read --records $R/records/$VIEW/$NAME/records.jsonl --url http://127.0.0.1:$PORT/v1 --model $MODEL --max-tokens $MAX --workers $W --out $ans/answers.jsonl ${EXTRA[@]+"${EXTRA[@]}"} > $ans/run.log 2>&1
  code=$?; [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "$VIEW $NAME exit $code seconds $(( $(date +%s) - T1 )) $(tail -1 $ans/run.log | cut -c1-140)"
  if [ $VIEW = locomo ]; then
    $PY -m studies.memory_bench.run score --answers $ans/answers.jsonl --data $LOC_DATA --split $LOC_SPLIT --part $PART --audit $AUDIT --out $ans/scores.jsonl > $ans/score.log 2>&1
    code=$?; [ $code = 0 ] || FAILED=$((FAILED + 1))
    tail -1 $ans/score.log | cut -c1-260 | sed "s/^/   /"
  fi
done < <(echo "$LIST" | grep -E "$ONLY" | grep -vE "$NOT")
metrics $JOB
kill $PID $SMI 2>/dev/null; wait $PID 2>/dev/null
echo "end $(date -u +%FT%TZ) rows failed $FAILED"
[ $FAILED = 0 ]
