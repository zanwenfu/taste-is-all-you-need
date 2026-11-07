#!/bin/bash
# Says whether a job has left the queue: exits 0 only on a plain answer that it has. A controller that cannot be reached, or no squeue at all, says nothing, and that is not an answer.
JOB=${1:?the job}
OUT=$(squeue -h -j "$JOB" -o %T 2>&1)
CODE=$?
if [ $CODE = 0 ]; then
  [ -z "$OUT" ]
else
  case "$OUT" in
    *"Invalid job id"*) exit 0;;
    *) exit 1;;
  esac
fi
