#!/bin/bash
#SBATCH --job-name=aos-long-serve
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
# One reader's server, the same settings as seed.sbatch's own, for several seeds to share as steps with SERVER_URL, so that no seed waits in the queue or loads the model again. READER is
# gpt-oss or laguna, which takes two GPUs and not on a node where a two-GPU server stalls. Free, no key. Ended by hand, by its job id, when its work is read. Writes its address to URL.
set -uo pipefail
unset ANTHROPIC_API_KEY OPENAI_API_KEY
D=${STUDY_ROOT:?}; READER=${READER:-laguna}; IMG=$D/images/vllm-v0.32.0.sif; OUT=$D/logs/long-serve-$SLURM_JOB_ID; mkdir -p $OUT
export APPTAINER_CACHEDIR=$D/cache/apptainer APPTAINER_TMPDIR=$D/tmp HF_HOME=$D/cache/hf HF_HUB_OFFLINE=1 TIKTOKEN_RS_CACHE_DIR=$D/cache/tiktoken VLLM_CACHE_ROOT=$D/cache/vllm XDG_CACHE_HOME=$D/cache/xdg PYTHONNOUSERSITE=1
if [ $READER = laguna ]; then
  FOLDER=$D/models/laguna-s-2.1-nvfp4; MODEL=poolside/laguna-s-2.1
  SERVE=(--tensor-parallel-size 2 --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --default-chat-template-kwargs '{"enable_thinking": true}')
else
  FOLDER=$D/models/gpt-oss-120b; MODEL=openai/gpt-oss-120b
  SERVE=(--tool-call-parser openai)
fi
echo "start $(date -u +%FT%TZ) on $(hostname) reader $READER"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 60 > $OUT/gpu.csv &
SMI=$!
PORT=$((20000 + RANDOM % 20000)); T0=$(date +%s)
apptainer exec --nv --home $D/cache/home --bind $D $IMG vllm serve $FOLDER --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
  --max-model-len 131072 --gpu-memory-utilization 0.92 --enable-auto-tool-choice "${SERVE[@]}" > $OUT/vllm.log 2>&1 &
PID=$!
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
  sleep 5; kill -0 $PID 2>/dev/null || { tail -40 $OUT/vllm.log; kill ${SMI:-} 2>/dev/null; echo "FAILED: server died"; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && { kill ${PID:-} ${SMI:-} 2>/dev/null; echo "FAILED: load timeout"; exit 1; }
done
echo "load_seconds $(( $(date +%s) - T0 )) port $PORT"
echo "http://127.0.0.1:$PORT/v1" > $OUT/url
wait $PID
kill ${SMI:-} 2>/dev/null
echo "end $(date -u +%FT%TZ)"
