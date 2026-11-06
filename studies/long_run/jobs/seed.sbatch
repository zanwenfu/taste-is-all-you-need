#!/bin/bash
#SBATCH --job-name=aos-long-seed
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# The long run: one seed per job, TASKS public tasks in one conversation, each arm side by side on this job's own server, each session read before more are run. Real task
# machines, each with its own process list, from the images fetched once. READER is gpt-oss or laguna, which takes two GPUs. Free, no key. Needs STUDY_ROOT, SHA and SEED.
set -uo pipefail
unset ANTHROPIC_API_KEY OPENAI_API_KEY
D=${STUDY_ROOT:?}; S=${SHA:?}; SEED=${SEED:?}; TASKS=${TASKS:-16}; READER=${READER:-gpt-oss}; PROMPT=${PROMPT:-anthropic}; COMPACT_AT=${COMPACT_AT:-8000}; KEEP=${KEEP:-2000}; TASK_CALLS=${TASK_CALLS:-40}
ARMS=${ARMS:-compaction taste taste_exists taste_check taste_search taste_rules notes_told notes_neutral restated no_compaction}
L=$D/long
IMG=$D/images/vllm-v0.32.0.sif
CODE=$D/code-${S:0:12}
export APPTAINER_CACHEDIR=$D/cache/apptainer APPTAINER_TMPDIR=$D/tmp XDG_CACHE_HOME=$D/cache/xdg HF_HOME=$D/cache/hf HF_HUB_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR=$D/cache/tiktoken VLLM_CACHE_ROOT=$D/cache/vllm PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export TASTE_COMMIT=$S MSWEA_SILENT_STARTUP=1 MSWEA_GLOBAL_CONFIG_DIR=$L/config
export GIT_AUTHOR_NAME=long-run GIT_AUTHOR_EMAIL=long-run@localhost GIT_COMMITTER_NAME=long-run GIT_COMMITTER_EMAIL=long-run@localhost
OUT=$L/runs/$READER-$PROMPT-seed$SEED-$SLURM_JOB_ID; mkdir -p $OUT
# Machines hold memory on a shared node: all of a seed's sit in one folder, gone with it however it ends. Seeds sharing a job's server each have their own.
WORK=/tmp/aos-long-$SLURM_JOB_ID-$READER-$PROMPT-seed$SEED; mkdir -p $WORK/machines
# When the job is stopped, each session gets the signal too and writes its record: this waits for them before their folder goes.
trap 'rm -rf "$WORK"' EXIT
trap 'kill ${PID:-} ${SMI:-} 2>/dev/null; wait; exit 143' TERM
if [ $READER = laguna ]; then
  # It thinks before it answers, and a thought counts toward the output cap: at 8,192 a hard task lost three replies in a row to the cap, and so left its machine, in 8 of 10 arms of the first session.
  FOLDER=$D/models/laguna-s-2.1-nvfp4; MODEL=poolside/laguna-s-2.1; MOST=16384
  SERVE=(--tensor-parallel-size 2 --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --default-chat-template-kwargs '{"enable_thinking": true}')
else
  FOLDER=$D/models/gpt-oss-120b; MODEL=openai/gpt-oss-120b; MOST=4096
  SERVE=(--tool-call-parser openai)
fi
fail() { echo "FAILED: $*"; kill ${PID:-} ${SMI:-} 2>/dev/null; exit 1; }
echo "start $(date -u +%FT%TZ) on $(hostname) code $S reader $READER prompt $PROMPT seed $SEED tasks $TASKS arms $ARMS"; df -h /tmp | tail -1
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 60 > $OUT/gpu.csv &
SMI=$!
# SERVER_URL: use a server another of this study's jobs is already running with this reader and these settings, as the memory cell can.
if [ -n "${SERVER_URL:-}" ]; then
  URL=$SERVER_URL; PID=
  curl -sf ${URL%/v1}/health >/dev/null || fail "no server at $URL"
  echo "using the running server at $URL"
else
  PORT=$((20000 + RANDOM % 20000)); URL=http://127.0.0.1:$PORT/v1; T0=$(date +%s)
  apptainer exec --nv --home $D/cache/home --bind $D $IMG vllm serve $FOLDER --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
    --max-model-len 131072 --gpu-memory-utilization 0.92 --enable-auto-tool-choice "${SERVE[@]}" > $OUT/vllm.log 2>&1 &
  PID=$!
  until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
    sleep 5; kill -0 $PID 2>/dev/null || { tail -40 $OUT/vllm.log; fail "server died"; }
    [ $(( $(date +%s) - T0 )) -gt 2400 ] && fail "load timeout"
  done
  echo "load_seconds $(( $(date +%s) - T0 ))"
fi
cd $CODE
$L/venv/bin/python -m studies.long_run.make_session --tasks-root $L/tb/terminal-bench-2-1 --agent-config $(ls $L/venv/lib/python3*/site-packages/minisweagent/config/mini.yaml) \
  --seed $SEED --tasks $TASKS --compact-at $COMPACT_AT --keep-recent $KEEP --prompt $PROMPT --later-rule-at 20 --task-calls $TASK_CALLS --images $L/images-sif --stop-at $((131072 - MOST - 8000)) --out $OUT/session.json | cut -c1-400
RUNS=()
for ARM in $ARMS; do
  ( trap '' TERM  # this shell stays until its session has written its record; the session sets its own handler
    T1=$(date +%s)
    TMPDIR=$WORK/machines $L/venv/bin/python -m studies.long_run.session --session $OUT/session.json --arm $ARM --env machine --url $URL --model $MODEL \
      --tokenizer $FOLDER/tokenizer.json --out $OUT/$ARM --workdir $WORK/$ARM --command-timeout 300 --max-tokens $MOST --writing-tokens $((2 * MOST)) > $OUT/$ARM.log 2>&1
    code=$?
    echo "$ARM exit $code seconds $(( $(date +%s) - T1 ))"; tail -1 $OUT/$ARM.log | cut -c1-900
    rm -rf $WORK/$ARM
    exit $code ) &
  RUNS+=($!)
done
FAILED=0
for RUN in "${RUNS[@]}"; do wait $RUN || FAILED=$((FAILED + 1)); done
curl -s ${URL%/v1}/metrics | grep -E "^vllm:(prompt_tokens_total|generation_tokens_total|prefix_cache_(hits|queries)_total|request_success_total)" > $OUT/vllm-metrics.txt
kill $PID $SMI 2>/dev/null; [ -n "$PID" ] && wait $PID 2>/dev/null
echo "end $(date -u +%FT%TZ) arms failed $FAILED"
[ $FAILED = 0 ]
