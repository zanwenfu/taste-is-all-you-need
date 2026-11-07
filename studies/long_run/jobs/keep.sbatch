#!/bin/bash
#SBATCH --job-name=aos-gpus
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
# The study's GPUs through the days the queue is full, since a two-GPU job can wait for days and the real test has its hour. Work reaches it as task files, run here on the node, so that no
# login node or laptop has to stay up: give_task.sh drops one in, even before the job starts, and a file NAME.stop beside a running task stops that task. A task named server-* is a server and
# does not count as work, and one reader can follow another on the same GPUs. It is not left idle: it ends itself, by its own id, after IDLE_SECONDS with its GPUs at rest and only servers
# running, or after HUNG_SECONDS with its GPUs at rest whatever says it is running. The price of ending is the GPUs, which a later job must queue for again. Free, no key.
set -uo pipefail
D=${STUDY_ROOT:?}; OUT=$D/logs/gpus-$SLURM_JOB_ID; TASKS=$OUT/tasks; mkdir -p $TASKS
IDLE_SECONDS=${IDLE_SECONDS:-3600}; HUNG_SECONDS=${HUNG_SECONDS:-28800}; BUSY_PERCENT=${BUSY_PERCENT:-5}; TICK=${TICK:-20}
echo "start $(date -u +%FT%TZ) on $(hostname), ends itself after $IDLE_SECONDS seconds idle or $HUNG_SECONDS with its GPUs at rest"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 60 > $OUT/gpu.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null; wait; exit 143' TERM

start_tasks() {
  local task name
  for task in "$TASKS"/*.sh; do
    [ -e "$task" ] || continue
    name=${task%.sh}
    [ -e "$name.pid" ] && continue
    # Each in a session of its own, so that stopping it reaches all it started and nothing else. The wrapper outlives a stop to write how the task ended.
    setsid bash -c 'trap : TERM; bash "$1.sh" > "$1.out" 2>&1; echo $? > "$1.exit"' _ "$name" &
    echo $! > "$name.pid"
    echo "task $(basename "$name") started $(date -u +%FT%TZ)"
  done
}

stop_tasks() {
  local stop name
  for stop in "$TASKS"/*.stop; do
    [ -e "$stop" ] || continue
    name=${stop%.stop}
    [ -e "$name.pid" ] && [ ! -e "$name.exit" ] && [ ! -e "$name.stopping" ] || continue
    # A task just started may not have its session yet, and then the next tick tells it. It is told once, since a task may be writing its record as it stops.
    kill -TERM -- "-$(cat "$name.pid")" 2>/dev/null || continue
    touch "$name.stopping"
    echo "task $(basename "$name") told to stop $(date -u +%FT%TZ)"
  done
}

working() {  # a task still running that is not a server
  local pid name
  for pid in "$TASKS"/*.pid; do
    [ -e "$pid" ] || continue
    name=${pid%.pid}
    case $(basename "$name") in server-*) continue;; esac
    [ -e "$name.exit" ] || return 0
  done
  return 1
}

gpu_busy() {  # also when nvidia-smi cannot say: a job is not ended on no evidence
  local most
  most=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  case "$most" in ''|*[!0-9]*) [ -n "${SAID_BLIND:-}" ] || echo "nvidia-smi cannot say how busy the GPUs are $(date -u +%FT%TZ)"; SAID_BLIND=1; return 0;; esac
  [ "$most" -gt "$BUSY_PERCENT" ]
}

end_job() {
  echo "ending: $1 $(date -u +%FT%TZ)"
  kill $SMI 2>/dev/null
  scancel $SLURM_JOB_ID
  exit 0
}

GPU_AT=$(date +%s); WORK_AT=$GPU_AT
while true; do
  start_tasks
  stop_tasks
  NOW=$(date +%s)
  gpu_busy && GPU_AT=$NOW
  working && WORK_AT=$NOW
  LAST=$(( GPU_AT > WORK_AT ? GPU_AT : WORK_AT ))
  [ $(( NOW - LAST )) -ge $IDLE_SECONDS ] && end_job "its GPUs at rest and no work for $IDLE_SECONDS seconds"
  [ $(( NOW - GPU_AT )) -ge $HUNG_SECONDS ] && end_job "its GPUs at rest for $HUNG_SECONDS seconds, whatever says it is running"
  sleep $TICK &
  wait $!
done
