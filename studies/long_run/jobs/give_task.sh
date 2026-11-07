#!/bin/bash
# Hand a kept job a task, whether or not the job has started: give_task.sh <job> <name> with the task's script on standard input. Written whole, so that the job never runs half a file, and
# never over one of the same name, since a task runs once. A name starting server- is a server, which does not count as work. Needs STUDY_ROOT.
set -euo pipefail
JOB=${1:?the job}; NAME=${2:?the name of the task}; TASKS=${STUDY_ROOT:?}/logs/gpus-$JOB/tasks
mkdir -p "$TASKS"
[ ! -e "$TASKS/$NAME.sh" ] || { echo "job $JOB already has a task $NAME" >&2; exit 1; }
cat > "$TASKS/$NAME.writing"
if [ ! -s "$TASKS/$NAME.writing" ]; then
  echo "an empty task" >&2
  mv "$TASKS/$NAME.writing" "$TASKS/$NAME.empty"
  exit 1
fi
mv "$TASKS/$NAME.writing" "$TASKS/$NAME.sh"
echo "job $JOB has task $NAME"
