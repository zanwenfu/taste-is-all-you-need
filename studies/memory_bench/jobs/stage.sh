#!/bin/bash
# One stage of a part as a task on a kept job: stage.sh <name> <its server's reader, or none> <job script> "<the stages it follows>". It waits for the part's hour (OPENS_EPOCH, on the held-out
# part) and for those stages, runs the job script on this job's server, and leaves a mark saying how it ended, so that the stages keep the order of open with nothing waiting on a login node. A
# stage it follows must have ended well, or only have ended if its name carries a ?. A stage that follows a failed one fails too, without running, as does one that follows a stage whose job has
# left the queue with no mark, or that has waited STAGE_WAIT_SECONDS. While it runs, NAME.running holds its job, so that a run again does not ask twice for it. Free.
set -uo pipefail
NAME=${1:?}; SERVER=${2:?}; SCRIPT=${3:?}; AFTER=${4:-}; D=${STUDY_ROOT:?}; MARKS=$D/bench/runs/${PART:?}-${RUN:?}/stages; CODE=$D/code-${SHA:0:12}/studies; POLL=${STAGE_POLL:-20}; LIMIT=${STAGE_WAIT_SECONDS:-172800}
mkdir -p "$MARKS"
mark() {
  echo "${SHA}" > "${MARKS:?}/${NAME:?}.$1"
  rm -f "${MARKS:?}/${NAME:?}.running"
  echo "$NAME $1 $(date -u +%FT%TZ) ${2:-}"
}
trap 'mark failed "told to stop"; exit 143' TERM
echo "$SLURM_JOB_ID" > "$MARKS/$NAME.running"
T0=$(date +%s)
if [ -n "${OPENS_EPOCH:-}" ]; then
  until [ "$(date +%s)" -ge "$OPENS_EPOCH" ]; do sleep "$POLL" & wait $!; done
fi
read -r -a WANTED <<< "$AFTER"
for STAGE in ${WANTED[@]+"${WANTED[@]}"}; do
  ANY=0
  case "$STAGE" in *\?) ANY=1; STAGE=${STAGE%\?};; esac
  until [ -e "$MARKS/$STAGE.ok" ] || [ -e "$MARKS/$STAGE.failed" ]; do
    if [ -e "$MARKS/$STAGE.running" ] && bash "$CODE/long_run/jobs/job_gone.sh" "$(cat "$MARKS/$STAGE.running")"; then
      [ $ANY = 1 ] && break
      mark failed "since $STAGE's job left the queue with no mark"
      exit 1
    fi
    if [ $(( $(date +%s) - T0 )) -ge "$LIMIT" ]; then
      mark failed "after $LIMIT seconds waiting for $STAGE"
      exit 1
    fi
    sleep "$POLL" & wait $!
  done
  if [ $ANY = 0 ] && [ ! -e "$MARKS/$STAGE.ok" ]; then
    mark failed "since $STAGE failed"
    exit 1
  fi
done
if [ "$SERVER" != none ]; then
  URL=$(bash "$CODE/long_run/jobs/server_url.sh" "$SLURM_JOB_ID" "$SERVER") || { mark failed "no $SERVER server"; exit 1; }
  export SERVER_URL=$URL
fi
bash "$CODE/memory_bench/jobs/$SCRIPT"
ENDED=$?
if [ $ENDED = 0 ]; then mark ok; else mark failed "exit $ENDED"; fi
exit $ENDED
