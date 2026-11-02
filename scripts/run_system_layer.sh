#!/bin/sh
# Start both arms of the system-layer study together (docs/studies/system-layer.md).
#
#   sudo scripts/run_system_layer.sh pilot   three tuning tasks, one trial per arm
#   sudo scripts/run_system_layer.sh full    the 69 test tasks, three trials per arm
#
# The paths are those of the study's machine; set them in the environment for another.
set -eu
MODE=${1:?pilot or full}
CODE=$(cd "$(dirname "$0")/.." && pwd)
export DATASET=${DATASET:-/data/tb/terminal-bench-2-1} SELECTED=${SELECTED:-/data/tb/runs} JOBS=${JOBS:-/data/tb/jobs}
export SECRETS=${SECRETS:-/root/taste-secrets/bedrock.env} MODEL=${MODEL:-azure/gpt-6-luna}
export TASTE_TRIALS_ROOT=${TASTE_TRIALS_ROOT:-/data/taste-trials}
export TASTE_CHECKPOINTS_ROOT=${TASTE_CHECKPOINTS_ROOT:-/data/taste-checkpoints}

CAPS="--ak agent=mini-swe-agent --ak spend_cap_usd=2 --ak worker_spend_cap_usd=2 --ak monitor_spend_cap_usd=1
      --ak worker_max_calls=1000 --ak monitor_max_calls=400 --ak max_commands=2000"
ALONE="--ak services=none --ak reply_reserve_seconds=10"
SUPERVISED="--ak rollback=on --ak keep_checkpoints=on --ak final_checkpoint=sl1 --ak checkpoint_mib=4096"

case $MODE in
  pilot) PART="tuning"; PREFIX=sl1-pilot; TRIALS="-k 1 -n 3 -i adaptive-rejection-sampler -i kv-store-grpc -i polyglot-c-py";;
  full)  PART="test system-layer-1"; PREFIX=sl1; TRIALS="-k 3 -n ${CONCURRENT:-10}";;
  *) echo "pilot or full" >&2; exit 2;;
esac

# shellcheck disable=SC2086  # the option lists split on purpose
"$CODE/infra/azure/run-terminal-bench.sh" "$PREFIX-alone" $PART $TRIALS --max-retries 2 $CAPS $ALONE
"$CODE/infra/azure/run-terminal-bench.sh" "$PREFIX-supervised" $PART $TRIALS --max-retries 2 $CAPS $SUPERVISED
