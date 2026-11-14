#!/bin/bash
#SBATCH --job-name=ap-replay
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:45:00
# Recorded trials rebuilt and graded in Apptainer on one compute node: no model call, no key, no Docker, no root.
#
#   ROOT=/usr/xtmp/<user>/<folder> CODE=$ROOT/code/<sha> TOKENS="<trial> ..." RESULTS=$ROOT/results/<name> \
#     sbatch --export=ALL --output=$ROOT/logs/%x-%j.out infra/duke/apptainer-replay.sbatch
#
# ROOT holds scripts/<trial>.json (replay scripts), recorded.json (what each Docker run recorded) and
# swebench-pro-os/v2/tasks (the public task folders). SIFs are kept in ROOT/sif and image layers in ROOT/cache;
# overlays live on node-local /tmp (tmpfs here, so they count against the job's memory) and are removed at the end.
set -euo pipefail
: "${ROOT:?}" "${CODE:?}" "${TOKENS:?}" "${RESULTS:?}"
export APPTAINER_CACHEDIR=$ROOT/cache
WORK=/tmp/ap-replay-$SLURM_JOB_ID
export APPTAINER_TMPDIR=$WORK/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RESULTS" "$ROOT/sif"
trap 'chmod -R u+rwX "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT
echo "host $(hostname) cpus $(nproc) start $(date '+%F %T %Z') code $CODE"
apptainer --version
status=0
for token in $TOKENS; do
  task=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["task"])' \
         "$ROOT/recorded.json" "$token")
  echo "== $token $task $(date '+%T')"
  python3 -I "$CODE/scripts/apptainer_replay.py" --script "$ROOT/scripts/$token.json" \
    --recorded "$ROOT/recorded.json" --token "$token" --task-dir "$ROOT/swebench-pro-os/v2/tasks/$task" \
    --sif-dir "$ROOT/sif" --work "$WORK" --out "$RESULTS/$token.json" ${EXTRA:-} || status=1
done
echo "end $(date '+%F %T %Z') status $status"
exit $status
