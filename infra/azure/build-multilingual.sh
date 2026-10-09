#!/bin/bash
# Build the pinned images of SWE-bench Multilingual tasks prepared by scripts/prepare_multilingual.py,
# $2 at a time (default 2: Docker stays responsive for running trials), stopping (without starting new
# builds) when the Docker disk has less than MIN_FREE_GB free. Each line of <prepared>/images.txt is
# "task base-image pinned-image"; a pinned image already present is kept as it is, never rebuilt.
#
#   sudo infra/azure/build-multilingual.sh /root/recovery-bench/runs/ml-w1 [parallel] [tasks-file]
#
# One line per task: "HH:MM:SS ok Ns task", "... FAIL task", "... SKIP low-disk task" or
# "... kept task"; <prepared>/built.tsv gets the task, the base image's digest and the image's id.
PREPARED=$1; PAR=${2:-2}; ONLY=${3:-}
DOCKER_DISK=${DOCKER_DISK:-/data}; MIN_FREE_GB=${MIN_FREE_GB:-150}
[ -f "$PREPARED/images.txt" ] || { echo "no $PREPARED/images.txt: run scripts/prepare_multilingual.py first" >&2; exit 2; }
build() {
  task=$1 base=$2 image=$3
  if [ -n "$ONLY" ] && ! grep -q -x -F "$task" "$ONLY"; then return; fi
  if docker image inspect "$image" > /dev/null 2>&1; then echo "$(date -u +%H:%M:%S) kept $task"; return; fi
  free=$(df --output=avail -BG "$DOCKER_DISK" | tail -1 | tr -dc 0-9)
  [ "$free" -lt "$MIN_FREE_GB" ] && { echo "$(date -u +%H:%M:%S) SKIP low-disk $task"; return; }
  s=$(date +%s)
  if docker pull -q "$base" > /dev/null 2>&1 \
      && docker build -q -t "$image" "$PREPARED/$task/environment" > /dev/null 2>"$PREPARED/$task.build.err"; then
    digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$base" 2>/dev/null)
    printf '%s\t%s\t%s\n' "$task" "$digest" "$(docker image inspect --format '{{.Id}}' "$image")" >> "$PREPARED/built.tsv"
    rm -f "$PREPARED/$task.build.err"
    echo "$(date -u +%H:%M:%S) ok $(( $(date +%s) - s ))s $task"
  else
    echo "$(date -u +%H:%M:%S) FAIL $task"
  fi
}
export -f build; export PREPARED ONLY DOCKER_DISK MIN_FREE_GB
xargs -a "$PREPARED/images.txt" -P "$PAR" -L 1 bash -c 'build "$@"' _
echo "$(date -u +%H:%M:%S) DONE $(df -h "$DOCKER_DISK" | tail -1)"
