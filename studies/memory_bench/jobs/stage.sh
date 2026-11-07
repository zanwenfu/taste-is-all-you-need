#!/bin/bash
# One stage of a part as a task on a kept job: stage.sh <name> <its server's reader, or none> <job script> "<the stages it follows>". It waits for the part's hour (OPENS_EPOCH, on the held-out
# part) and for those stages, runs the job script on this job's server, and leaves a mark saying how it ended, so that the stages keep the order of open with nothing waiting on a login node. A
# stage it follows must have ended well, or only have ended if its name carries a ?. A stage that follows a failed one fails too, without running, as does one that follows a stage whose job has
# left the queue with no mark, or that has waited STAGE_WAIT_SECONDS. While it runs, NAME.running holds its job, so that a run again does not ask twice for it, and NAME.beat the time it last said
# it was alive, every STAGE_BEAT seconds, so that a stage killed outright on a job still up is known by its silence: what follows it fails once the silence passes STAGE_SILENT_SECONDS and still
# holds STAGE_CONFIRM_SECONDS later, as a stalled file system that stops every node a while must not end anything. Free.
set -uo pipefail
NAME=${1:?}; SERVER=${2:?}; SCRIPT=${3:?}; AFTER=${4:-}; D=${STUDY_ROOT:?}; MARKS=$D/bench/runs/${PART:?}-${RUN:?}/stages; CODE=$D/code-${SHA:0:12}/studies; POLL=${STAGE_POLL:-20}; LIMIT=${STAGE_WAIT_SECONDS:-172800}
BEAT=${STAGE_BEAT:-60}; SILENT=${STAGE_SILENT_SECONDS:-1800}; CONFIRM=${STAGE_CONFIRM_SECONDS:-180}
mkdir -p "$MARKS"
mark() {
  if [ -n "${BEATING:-}" ]; then kill "$BEATING" 2>/dev/null; wait "$BEATING" 2>/dev/null; fi
  echo "${SHA}" > "${MARKS:?}/${NAME:?}.$1"
  rm -f "${MARKS:?}/${NAME:?}.running" "${MARKS:?}/${NAME:?}.beat"
  echo "$NAME $1 $(date -u +%FT%TZ) ${2:-}"
}
beat() { echo "$(date +%s)" > "$MARKS/$NAME.beat"; }
quiet() {  # STAGE: seconds since it last said it was alive, or nothing from a stage on code from before the beat, or between its writes
  local at; at=$(cat "$MARKS/$1.beat" 2>/dev/null)
  case "$at" in ''|*[!0-9]*) return 0;; esac
  echo $(( $(date +%s) - at ))
}
trap 'mark failed "told to stop"; exit 143' TERM
beat
echo "$SLURM_JOB_ID" > "$MARKS/$NAME.running"
# Its own process, and silent once this shell is gone however it went. Its output goes nowhere, so that nothing waiting on this shell's output waits on it.
( while sleep "$BEAT"; kill -0 $$ 2>/dev/null; do beat; done ) < /dev/null > /dev/null 2>&1 &
BEATING=$!
T0=$(date +%s)
if [ -n "${OPENS_EPOCH:-}" ]; then
  until [ "$(date +%s)" -ge "$OPENS_EPOCH" ]; do sleep "$POLL" & wait $!; done
fi
read -r -a WANTED <<< "$AFTER"
for STAGE in ${WANTED[@]+"${WANTED[@]}"}; do
  ANY=0; HUSHED=
  case "$STAGE" in *\?) ANY=1; STAGE=${STAGE%\?};; esac
  until [ -e "$MARKS/$STAGE.ok" ] || [ -e "$MARKS/$STAGE.failed" ]; do
    WHY=
    if [ -e "$MARKS/$STAGE.running" ]; then
      if bash "$CODE/long_run/jobs/job_gone.sh" "$(cat "$MARKS/$STAGE.running")"; then
        WHY="since $STAGE's job left the queue with no mark"
      else
        QUIET=$(quiet "$STAGE")
        if [ -n "$QUIET" ] && [ "$QUIET" -ge "$SILENT" ]; then
          [ -n "$HUSHED" ] || HUSHED=$(date +%s)
          [ $(( $(date +%s) - HUSHED )) -ge "$CONFIRM" ] && WHY="since $STAGE has said nothing for $QUIET seconds, with no mark, on job $(cat "$MARKS/$STAGE.running" 2>/dev/null)"
        else
          HUSHED=
        fi
      fi
    fi
    # A stage leaves its mark before it clears up, so a mark seen now means it ended between the looks.
    if [ -n "$WHY" ] && [ ! -e "$MARKS/$STAGE.ok" ] && [ ! -e "$MARKS/$STAGE.failed" ]; then
      [ $ANY = 1 ] && break
      mark failed "$WHY"
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
