#!/bin/sh
# Run one part of a registered Terminal-Bench split with the Taste agent.
#
#   sudo infra/azure/run-terminal-bench.sh <job-name> tuning [harbor run options...]
#   sudo infra/azure/run-terminal-bench.sh <job-name> test <study> [harbor run options...]
#
# The part's tasks are copied from the downloaded dataset into a new directory
# by taste.benchmarks.task_split, which refuses a changed dataset, and refuses
# test tasks unless <study> is registered in data/studies/<study>.json for this
# exact split. The variables can be overridden in the environment:
#   SPLIT      the split record (default data/splits/terminal-bench-2-1.json)
#   DATASET    the downloaded dataset (default /root/tb/terminal-bench-2-1)
#   SELECTED   where selections are made (default /root/tb/runs)
#   JOBS       where Harbor writes jobs (default /root/tb/jobs)
# Everything else is as for run-harbor.sh, which runs the job.
set -eu

[ "$(id -u)" = 0 ] || { echo "run as root: the trial owner needs Docker and systemd" >&2; exit 1; }
[ $# -ge 2 ] || { sed -n 2,14p "$0" >&2; exit 2; }
JOB=$1; PART=$2; shift 2
case $JOB in *[!A-Za-z0-9_-]*|"") echo "job name: letters, digits, - and _ only" >&2; exit 2;; esac
STUDY=""
case $PART in
  tuning) ;;
  test)
    [ $# -ge 1 ] || { echo "test tasks run only for a registered study" >&2; exit 2; }
    STUDY=$1; shift
    case $STUDY in *[!A-Za-z0-9_-]*|"") echo "study: letters, digits, - and _ only" >&2; exit 2;; esac;;
  *) echo "part: tuning or test" >&2; exit 2;;
esac

SOURCE=$(cd "$(dirname "$0")/../.." && pwd)
SPLIT=${SPLIT:-$SOURCE/data/splits/terminal-bench-2-1.json}
DATASET=${DATASET:-/root/tb/terminal-bench-2-1}
SELECTED=${SELECTED:-/root/tb/runs}
export JOBS=${JOBS:-/root/tb/jobs}

mkdir -p "$SELECTED"
( cd "$SOURCE" && python3 -m taste.benchmarks.task_split select "$SPLIT" "$DATASET" "$PART" \
    "$SELECTED/$JOB" --registrations "$SOURCE/data/studies" ${STUDY:+--study "$STUDY"} )
exec "$SOURCE/infra/azure/run-harbor.sh" "$JOB" "$SELECTED/$JOB" "$@"
