#!/bin/bash
#SBATCH --job-name=aos-long-seed
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# The long run: one seed per job, TASKS public tasks in one conversation, each arm side by side on this job's own server, each session read before more are run. Real task
# containers, from the images fetched once. READER is gpt-oss or laguna, which takes two GPUs. Free, no key. Needs STUDY_ROOT, SHA and SEED.
set -uo pipefail
unset ANTHROPIC_API_KEY OPENAI_API_KEY
D=${STUDY_ROOT:?}; S=${SHA:?}; SEED=${SEED:?}; TASKS=${TASKS:-16}; READER=${READER:-gpt-oss}; PROMPT=${PROMPT:-anthropic}; COMPACT_AT=${COMPACT_AT:-8000}; KEEP=${KEEP:-2000}
ARMS=${ARMS:-compaction taste taste_exists taste_check taste_rules notes_told notes_neutral restated no_compaction}
L=$D/long
IMG=$D/images/vllm-v0.32.0.sif
CODE=$D/code-${S:0:12}
export APPTAINER_CACHEDIR=$D/cache/apptainer APPTAINER_TMPDIR=$D/tmp XDG_CACHE_HOME=$D/cache/xdg HF_HOME=$D/cache/hf HF_HUB_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR=$D/cache/tiktoken VLLM_CACHE_ROOT=$D/cache/vllm PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export TASTE_COMMIT=$S MSWEA_SILENT_STARTUP=1 MSWEA_GLOBAL_CONFIG_DIR=$L/config
export GIT_AUTHOR_NAME=long-run GIT_AUTHOR_EMAIL=long-run@localhost GIT_COMMITTER_NAME=long-run GIT_COMMITTER_EMAIL=long-run@localhost
OUT=$L/runs/$READER-$PROMPT-seed$SEED-$SLURM_JOB_ID; mkdir -p $OUT
if [ $READER = laguna ]; then
  FOLDER=$D/models/laguna-s-2.1-nvfp4; MODEL=poolside/laguna-s-2.1
  SERVE=(--tensor-parallel-size 2 --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --default-chat-template-kwargs '{"enable_thinking": true}')
else
  FOLDER=$D/models/gpt-oss-120b; MODEL=openai/gpt-oss-120b
  SERVE=(--tool-call-parser openai)
fi
fail() { echo "FAILED: $*"; kill ${PID:-0} ${SMI:-0} 2>/dev/null; exit 1; }
echo "start $(date -u +%FT%TZ) on $(hostname) code $S reader $READER prompt $PROMPT seed $SEED tasks $TASKS arms $ARMS"; df -h /tmp | tail -1
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 60 > $OUT/gpu.csv &
SMI=$!
PORT=$((20000 + RANDOM % 20000)); T0=$(date +%s)
apptainer exec --nv --home $D/cache/home --bind $D $IMG vllm serve $FOLDER --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
  --max-model-len 131072 --gpu-memory-utilization 0.92 --enable-auto-tool-choice "${SERVE[@]}" > $OUT/vllm.log 2>&1 &
PID=$!
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
  sleep 5; kill -0 $PID 2>/dev/null || { tail -40 $OUT/vllm.log; fail "server died"; }
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && fail "load timeout"
done
echo "load_seconds $(( $(date +%s) - T0 ))"
cd $CODE
$L/venv/bin/python -m studies.long_run.make_session --tasks-root $L/tb/terminal-bench-2-1 --agent-config $(ls $L/venv/lib/python3*/site-packages/minisweagent/config/mini.yaml) \
  --seed $SEED --tasks $TASKS --compact-at $COMPACT_AT --keep-recent $KEEP --prompt $PROMPT --later-rule-at 20 --task-calls 40 --images $L/images-sif --out $OUT/session.json | cut -c1-400
RUNS=()
for ARM in $ARMS; do
  ( T1=$(date +%s)
    $L/venv/bin/python -m studies.long_run.session --session $OUT/session.json --arm $ARM --env singularity --url http://127.0.0.1:$PORT/v1 --model $MODEL \
      --tokenizer $FOLDER/tokenizer.json --out $OUT/$ARM --workdir /tmp/aos-long-$SLURM_JOB_ID-$ARM --command-timeout 300 > $OUT/$ARM.log 2>&1
    echo "$ARM exit $? seconds $(( $(date +%s) - T1 ))"; tail -1 $OUT/$ARM.log | cut -c1-900
    rm -rf /tmp/aos-long-$SLURM_JOB_ID-$ARM ) &
  RUNS+=($!)
done
wait "${RUNS[@]}"
curl -s http://127.0.0.1:$PORT/metrics | grep -E "^vllm:(prompt_tokens_total|generation_tokens_total|prefix_cache_(hits|queries)_total|request_success_total)" > $OUT/vllm-metrics.txt
kill $PID $SMI 2>/dev/null; wait $PID 2>/dev/null
echo "end $(date -u +%FT%TZ)"
