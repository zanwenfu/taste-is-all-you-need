#!/bin/bash
#SBATCH --job-name=aos-probe-cell
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# One cell of the rule-loss matrix: a case set (IN), a summary prompt (PR) and a depth (K), every arm, on the settings frozen in the protocol. Notes and headers are written
# once per case set, in OUT, and any job can be run again and carries on. Free. Needs STUDY_ROOT and SHA.
set -uo pipefail
unset ANTHROPIC_API_KEY OPENAI_API_KEY
D=${STUDY_ROOT:?}; S=${SHA:?}
IMG=$D/images/vllm-v0.32.0.sif
CODE=$D/code-${S:0:12}
PR=${PR:?}; K=${K:?}; IN=${IN:?}; OUT=${OUT:?}; NOTES_FROM=${NOTES_FROM:-}; HEADERS_FROM=${HEADERS_FROM:-}; BUDGET=${BUDGET:-4096}; LABEL=${LABEL:-}; PROCS=${PROCS:-28}; FAILED=0
JOB=$OUT/job-$SLURM_JOB_ID; mkdir -p $JOB
export APPTAINER_CACHEDIR=$D/cache/apptainer APPTAINER_TMPDIR=$D/tmp HF_HOME=$D/cache/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR=$D/cache/tiktoken VLLM_CACHE_ROOT=$D/cache/vllm XDG_CACHE_HOME=$D/cache/xdg PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TASTE_COMMIT=$S
export GIT_AUTHOR_NAME=memory-probe GIT_AUTHOR_EMAIL=memory-probe@localhost GIT_COMMITTER_NAME=memory-probe GIT_COMMITTER_EMAIL=memory-probe@localhost
PY=$D/venv-probe/bin/python
RUN="apptainer exec --home $D/cache/home --bind $D $IMG"
MODEL=openai/gpt-oss-120b
fail() { echo "FAILED: $*"; kill ${PID:-0} ${SMI:-0} 2>/dev/null; exit 1; }
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT cell $PR k$K budget $BUDGET in $IN out $OUT"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 > $JOB/gpu.csv &
SMI=$!
PORT=$((20000 + RANDOM % 20000)); URL=http://127.0.0.1:$PORT/v1; T0=$(date +%s)
apptainer exec --nv --home $D/cache/home --bind $D $IMG vllm serve $D/models/gpt-oss-120b --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
  --max-model-len 131072 --gpu-memory-utilization 0.92 --enable-auto-tool-choice --tool-call-parser openai > $JOB/vllm.log 2>&1 &
PID=$!
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
  sleep 5; kill -0 $PID 2>/dev/null || { tail -40 $JOB/vllm.log; fail "server died"; }
  [ $(( $(date +%s) - T0 )) -gt 1800 ] && fail "load timeout"
done
echo "load_seconds $(( $(date +%s) - T0 ))"
cd $CODE
TOLD=$OUT/notes-told.jsonl; NEUTRAL=$OUT/notes-neutral.jsonl; HEADERS=$OUT/turn-headers.jsonl
# Notes written earlier carry over unless a version came back empty: those cases are written again.
if [ -n "$NOTES_FROM" ] && [ ! -e $TOLD ]; then
  $PY - $NOTES_FROM $OUT <<'PY'
import json, sys
from pathlib import Path
source, out = Path(sys.argv[1]), Path(sys.argv[2])
for name in ("notes-told.jsonl", "notes-neutral.jsonl"):
    rows = [json.loads(line) for line in (source / name).read_text().splitlines()]
    kept = [row for row in rows if all(version.strip() for version in row["versions"])]
    (out / name).write_text("".join(json.dumps(row) + "\n" for row in kept))
    print(name, "kept", len(kept), "of", len(rows))
PY
fi
[ -n "$HEADERS_FROM" ] && [ ! -e $HEADERS ] && cp $HEADERS_FROM/turn-headers.jsonl $HEADERS
for style in told neutral; do
  T1=$(date +%s)
  $PY -m studies.memory_probe.notes --inputs $IN --url $URL --model $MODEL --style $style --workers 32 --out $OUT/notes-$style.jsonl 2>&1 | tail -1 | cut -c1-200
  echo "notes $style seconds $(( $(date +%s) - T1 ))"
done
T1=$(date +%s)
$PY -m studies.memory_probe.headers --inputs $IN --url $URL --model $MODEL --out $HEADERS --workers 32 2>&1 | tail -1 | cut -c1-200
echo "headers seconds $(( $(date +%s) - T1 ))"
one() {  # label arm condition extra...
  local label=$1 arm=$2 condition=$3; shift 3
  local name=$arm-$condition-$PR-k$K-b$BUDGET T1=$(date +%s) row=$OUT/$label/$arm-$condition-$PR-k$K-b$BUDGET PIDS=() I code
  mkdir -p $row
  for I in $(seq 0 $((PROCS-1))); do
    $PY -m studies.memory_probe.run --inputs $IN --prompt $PR --k $K --arm $arm --condition $condition --budget $BUDGET --shard $I/$PROCS --workers 1 \
        --url $URL --model $MODEL --tokenizer $D/models/gpt-oss-120b/tokenizer.json \
        --workdir /tmp/aos-probe-$SLURM_JOB_ID-$label-$name --out $row/shard$I.jsonl "$@" > $row/shard$I.log 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  $PY -m studies.memory_probe.merge --inputs $IN --out $row/results.jsonl $row/shard*[0-9].jsonl > $row/run.log 2>&1
  code=$?; [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "$label $name exit $code seconds $(( $(date +%s) - T1 )) $(tail -1 $row/run.log | cut -c1-300)"
  rm -rf /tmp/aos-probe-$SLURM_JOB_ID-$label-$name
}
judge() {  # label arm
  local results=$OUT/$1/$2-i-$PR-k$K-b$BUDGET/results.jsonl
  echo "judge $1 $($RUN $D/venv/bin/python -m studies.memory_probe.judge --results $results --cases $IN/cases.jsonl --url $URL --model $MODEL 2>&1 | tail -1 | cut -c1-200)"
}
one summary a1 i
one restated a1 i --upper
one flat-newest a2a i
one rules a6 i
one rules-in-headers a6h i
one turns a5q i
one turns-headers a4q i --headers $HEADERS
one rules-and-search a7q i --headers $HEADERS
for style in told neutral; do
  N=$OUT/notes-$style.jsonl
  one notes-last-$style a2cq i --notes $N
  one notes-every-$style a2dq i --notes $N
  one notes-newest-$style a2cw i --notes $N
  one notes-newest-and-older-$style a2dw i --notes $N
done
if [ "${SMALL:-0}" = 0 ]; then
  one header-list a3 i --headers $HEADERS
  one plain-nothing a5 vi
  one plain-exists a5 vii
  one plain-check a5 viii
fi
judge turns a5q; judge turns-headers a4q
for style in told neutral; do
  judge notes-last-$style a2cq; judge notes-every-$style a2dq; judge notes-newest-$style a2cw; judge notes-newest-and-older-$style a2dw
done
curl -s http://127.0.0.1:$PORT/metrics | grep -E "^vllm:(prompt_tokens_total|generation_tokens_total|prefix_cache_(hits|queries)_total|request_success_total)" > $JOB/vllm-metrics.txt
kill $PID $SMI 2>/dev/null; wait $PID 2>/dev/null
echo "end $(date -u +%FT%TZ) rows failed $FAILED"
[ $FAILED = 0 ]
