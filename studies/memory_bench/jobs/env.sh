# What every stage of the comparison shares. Needs STUDY_ROOT, PART (dev or held_out) and SHA (the commit under test). No key.
unset ANTHROPIC_API_KEY OPENAI_API_KEY
PART=${PART:?dev or held_out}; SHA=${SHA:?the commit under test}; D=${STUDY_ROOT:?the folder that holds this study}
IMG=$D/images/vllm-v0.32.0.sif
B=$D/bench
S=${SHA:0:12}
CODE=$D/code-$S
R=$B/runs/$PART-$S
V=$R/views
[ -d "$CODE" ] || { echo "no code at $CODE"; exit 1; }
export APPTAINER_CACHEDIR=$D/cache/apptainer APPTAINER_TMPDIR=$D/tmp HF_HOME=$D/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
export TIKTOKEN_RS_CACHE_DIR=$D/cache/tiktoken VLLM_CACHE_ROOT=$D/cache/vllm XDG_CACHE_HOME=$D/cache/xdg PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TASTE_COMMIT=$SHA
export FASTEMBED_CACHE_PATH=$D/cache/fastembed MEM0_TELEMETRY=False ANONYMIZED_TELEMETRY=False OMP_NUM_THREADS=1
export GIT_AUTHOR_NAME=memory-bench GIT_AUTHOR_EMAIL=memory-bench@localhost GIT_COMMITTER_NAME=memory-bench GIT_COMMITTER_EMAIL=memory-bench@localhost
RUN="apptainer exec --home $D/cache/home --bind $D $IMG"
PY=$D/venv-probe/bin/python
LME_DATA=$B/data/longmemeval_s_cleaned.json
LOC_DATA=$B/src/locomo/data/locomo10.json
LME_SPLIT=$CODE/data/splits/memory-bench-lme.json
LOC_SPLIT=$CODE/data/splits/memory-bench-locomo.json
AUDIT=$B/src/locomo-audit/errors.json
JUDGE_PROMPTS=$B/src/mem0-benchmarks/run.py
LME_REF=$D/lme/data/longmemeval_oracle.json
LLAMA=$D/models/llama-3.1-70b-instruct
OSS=$D/models/gpt-oss-120b
OSS_NAME=openai/gpt-oss-120b
LLAMA_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
FROZEN=(); [ "$PART" = held_out ] && FROZEN=(--frozen-config $SHA)
fail() { echo "FAILED: $*"; kill ${PID:-0} ${EMB:-0} ${SMI:-0} 2>/dev/null; exit 1; }
gpu_log() { nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 60 > $1/gpu.csv & SMI=$!; }
serve_oss() {  # job folder, window, share of the GPU's memory; sets PORT and PID
  PORT=$((20000 + RANDOM % 20000)); local t0=$(date +%s)
  $RUN_GPU vllm serve $OSS --served-model-name $OSS_NAME --host 127.0.0.1 --port $PORT --max-model-len $2 --gpu-memory-utilization $3 > $1/vllm.log 2>&1 &
  PID=$!
  wait_for $1 $PORT $t0
}
wait_for() {  # job folder, port, start time
  until curl -sf http://127.0.0.1:$2/health >/dev/null; do
    sleep 5; kill -0 $PID 2>/dev/null || { tail -40 $1/vllm.log; fail "server died"; }
    [ $(( $(date +%s) - $3 )) -gt 2400 ] && fail "load timeout"
  done
  echo "load_seconds $(( $(date +%s) - $3 ))"
}
RUN_GPU="apptainer exec --nv --home $D/cache/home --bind $D $IMG"
metrics() { curl -s http://127.0.0.1:$PORT/metrics | grep -E "^vllm:(prompt_tokens_total|generation_tokens_total|prefix_cache_(hits|queries)_total|request_success_total)" > $1/vllm-metrics.txt; }
