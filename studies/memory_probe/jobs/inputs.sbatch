#!/bin/bash
#SBATCH --job-name=aos-probe-inputs
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:00:00
# A case set's own baselines (no rule, the summary alone, the rule restated) and its compaction records, on the first 10 histories, then the inputs the rows read. Moved,
# reversed and withdrawn cases need their own, since a later statement changes the history. A cued case keeps the published case's history, summaries, records and notes
# (its inputs are built apart), and needs from here only its own no-rule rate, which the analysis reads with --floors. Free. Needs STUDY_ROOT, SHA, VARIANT and OUT.
set -uo pipefail
unset ANTHROPIC_API_KEY OPENAI_API_KEY
D=${STUDY_ROOT:?}; S=${SHA:?}; V=${VARIANT:?}; OUT=${OUT:?}; PR=${PR:-anthropic}; HISTORIES=${HISTORIES:-0:10}
IMG=$D/images/vllm-v0.32.0.sif
CODE=$D/code-${S:0:12}
JOB=$OUT/job-$SLURM_JOB_ID; mkdir -p $JOB
export APPTAINER_CACHEDIR=$D/cache/apptainer APPTAINER_TMPDIR=$D/tmp HF_HOME=$D/cache/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR=$D/cache/tiktoken VLLM_CACHE_ROOT=$D/cache/vllm XDG_CACHE_HOME=$D/cache/xdg PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TASTE_COMMIT=$S
RUN="apptainer exec --home $D/cache/home --bind $D $IMG"
MODEL=openai/gpt-oss-120b
fail() { echo "FAILED: $*"; kill ${PID:-} ${SMI:-} 2>/dev/null; exit 1; }
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT variant $V prompt $PR histories $HISTORIES out $OUT"
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
printf -v NAME 'h%02d-%02d' ${HISTORIES%%:*} ${HISTORIES##*:}  # the name the export looks for
mkdir -p $OUT/full/hermes-$PR $OUT/repeated/$PR
T1=$(date +%s)
$RUN $D/venv/bin/python -m studies.context_rot.conpact_bridge --source hermes --histories $HISTORIES --variant $V --prompt $PR \
  --judge defer --probe-url $URL --workers 16 --out $OUT/full/hermes-$PR/$NAME > $OUT/full-$PR.report.json 2> $OUT/full-$PR.stderr.log || fail "baseline"
echo "baseline seconds $(( $(date +%s) - T1 ))"
if [ "$V" = cued ]; then
  kill $PID $SMI 2>/dev/null; wait $PID 2>/dev/null
  echo "end $(date -u +%FT%TZ)"
  exit 0
fi
T1=$(date +%s)
$RUN $D/venv/bin/python -m studies.context_rot.repeated --prompt $PR --records-only --histories $HISTORIES --variant $V --probe-url $URL \
  --tokenizer $D/models/gpt-oss-120b --workers 15 --out $OUT/repeated/$PR/$NAME > $OUT/repeated-$PR.stdout 2> $OUT/repeated-$PR.stderr.log || fail "records"
echo "records seconds $(( $(date +%s) - T1 )) $(tail -1 $OUT/repeated-$PR.stdout | cut -c1-200)"
# The inputs are one prompt's, so a second prompt's have a folder of their own.
INPUTS=$OUT/inputs; [ "$PR" = anthropic ] || INPUTS=$OUT/inputs-$PR
$RUN $D/venv/bin/python -m studies.memory_probe.export --full $OUT/full --repeated $OUT/repeated --histories $HISTORIES --variant $V --deep 0 --prompts $PR --out $INPUTS 2>&1 | tail -1 | cut -c1-200
[ -s $INPUTS/delivered.jsonl ] || fail "export"
curl -s http://127.0.0.1:$PORT/metrics | grep -E "^vllm:(prompt_tokens_total|generation_tokens_total|prefix_cache_(hits|queries)_total|request_success_total)" > $JOB/vllm-metrics.txt
kill $PID $SMI 2>/dev/null; wait $PID 2>/dev/null
echo "end $(date -u +%FT%TZ)"
