#!/bin/bash
# One stage of a part as a task on a kept job: stage.sh <name> <its server's reader, or none> <job script> "<the stages it follows>". It waits for the part's hour (OPENS_EPOCH, on the held-out
# part) and for those stages, runs the job script on this job's server, and leaves a mark saying how it ended, so that the stages keep the order of open with nothing waiting on a login node. A
# stage it follows must have ended well, or only have ended if its name carries a ?. A stage that follows a failed one fails too, without running. Free.
set -uo pipefail
NAME=${1:?}; SERVER=${2:?}; SCRIPT=${3:?}; AFTER=${4:-}; D=${STUDY_ROOT:?}; MARKS=$D/bench/runs/${PART:?}-${RUN:?}/stages; CODE=$D/code-${SHA:0:12}/studies; POLL=${STAGE_POLL:-20}
mkdir -p "$MARKS"
mark() { touch "$MARKS/$NAME.$1"; echo "$NAME $1 $(date -u +%FT%TZ) ${2:-}"; }
trap 'mark failed "told to stop"; exit 143' TERM
if [ -n "${OPENS_EPOCH:-}" ]; then
  until [ "$(date +%s)" -ge "$OPENS_EPOCH" ]; do sleep "$POLL" & wait $!; done
fi
read -r -a WANTED <<< "$AFTER"
for STAGE in ${WANTED[@]+"${WANTED[@]}"}; do
  ANY=0
  case "$STAGE" in *\?) ANY=1; STAGE=${STAGE%\?};; esac
  until [ -e "$MARKS/$STAGE.ok" ] || [ -e "$MARKS/$STAGE.failed" ]; do sleep "$POLL" & wait $!; done
  if [ ! -e "$MARKS/$STAGE.ok" ] && [ $ANY = 0 ]; then mark failed "since $STAGE failed"; exit 1; fi
done
if [ "$SERVER" != none ]; then
  URL=$(bash "$CODE/long_run/jobs/server_url.sh" "$SLURM_JOB_ID" "$SERVER") || { mark failed "no $SERVER server"; exit 1; }
  export SERVER_URL=$URL
fi
bash "$CODE/memory_bench/jobs/$SCRIPT"
ENDED=$?
if [ $ENDED = 0 ]; then mark ok; else mark failed "exit $ENDED"; fi
exit $ENDED
