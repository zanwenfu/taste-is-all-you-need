#!/bin/bash
# The address of a kept job's server once it has loaded: server_url.sh <job> <reader>. Fails if its task has ended, or at SERVER_WAIT seconds. Needs STUDY_ROOT.
set -uo pipefail
JOB=${1:?the job}; READER=${2:?the reader}; D=${STUDY_ROOT:?}; WAIT=${SERVER_WAIT:-2400}; URL=$D/logs/long-serve-$JOB-$READER/url; TASKS=$D/logs/gpus-$JOB/tasks; T0=$(date +%s)
ended() {  # its tasks have started and none is still running
  local seen=1 pid
  for pid in "$TASKS"/server-"$READER"*.pid; do
    [ -e "$pid" ] || continue
    seen=0
    [ -e "${pid%.pid}.exit" ] || return 1
  done
  return $seen
}
until [ -s "$URL" ]; do
  ended && { echo "the $READER server's task has ended on job $JOB" >&2; exit 1; }
  [ $(( $(date +%s) - T0 )) -ge "$WAIT" ] && { echo "no $READER server on job $JOB after $WAIT seconds" >&2; exit 1; }
  sleep "${SERVER_POLL:-5}"
done
cat "$URL"
