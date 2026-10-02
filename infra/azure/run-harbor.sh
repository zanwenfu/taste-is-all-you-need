#!/bin/sh
# Run a Harbor job with the Taste agent on this host, in one bounded unit.
#
#   sudo infra/azure/run-harbor.sh <job-name> <tasks-dir> [harbor run options...]
#
# Example, one errata-bench task, one attempt:
#   sudo infra/azure/run-harbor.sh dry1 /root/errata/v1.0.2-dataset/harbor \
#        -i 135yshr-savanna-vet-go-28 -k 1 -n 1
#
# Everything below is this host's wiring; nothing changes a task. The
# variables can be overridden in the environment:
#   TASTE_SOURCE   root-owned, read-only checkout of this repository
#   HARBOR_VENV    the Harbor environment (harbor 0.23.0)
#   WORKER_PYTHON  interpreter of the worker environment, run as the worker user
#   SECRETS        root-only file with AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_KEY
#   EXTRA_PATH     further entries for PYTHONPATH (a benchmark's own agents)
#   MODEL          the model every role uses unless --ak worker_model says otherwise
set -eu

[ "$(id -u)" = 0 ] || { echo "run as root: the trial owner needs Docker and systemd" >&2; exit 1; }
[ $# -ge 2 ] || { sed -n 2,9p "$0" >&2; exit 2; }
JOB=$1; TASKS=$2; shift 2

TASTE_SOURCE=${TASTE_SOURCE:-$(cd "$(dirname "$0")/../.." && pwd)}
HARBOR_VENV=${HARBOR_VENV:-/root/taste-harbor-20260927/venv}
WORKER_PYTHON=${WORKER_PYTHON:-/home/bugbash/taste-openai-20260923.venv/bin/python}
SECRETS=${SECRETS:-/root/taste-secrets/azure.env}
JOBS=${JOBS:-/root/errata/jobs}
MODEL=${MODEL:-azure/gpt-6-astra}
DOCKER_CONFIG=${DOCKER_CONFIG:-/root/taste-harbor-20260927/docker-config}
TRIALS=${TASTE_TRIALS_ROOT:-/var/lib/taste-trials}

case $JOB in *[!A-Za-z0-9_-]*|"") echo "job name: letters, digits, - and _ only" >&2; exit 2;; esac
# Name every unmet requirement now, rather than fail at the first trial.
if [ "${SKIP_HOST_CHECK:-0}" != 1 ]; then
  report=$(TASTE_SOURCE="$TASTE_SOURCE" HARBOR_VENV="$HARBOR_VENV" WORKER_PYTHON="$WORKER_PYTHON" \
           SECRETS="$SECRETS" DOCKER_CONFIG="$DOCKER_CONFIG" TASTE_TRIALS_ROOT="$TRIALS" \
           "$(dirname "$0")/check-host.sh") || { printf '%s\n' "$report" >&2; exit 1; }
fi
[ "$(stat -c %a "$SECRETS")" = 600 ] && [ "$(stat -c %u "$SECRETS")" = 0 ] \
  || { echo "$SECRETS must be a root-only file (mode 600)" >&2; exit 1; }
# The worker user reads the code but must not be able to change what root runs.
[ "$(stat -c %u "$TASTE_SOURCE")" = 0 ] || { echo "$TASTE_SOURCE must be owned by root" >&2; exit 1; }
[ ! -e "$JOBS/$JOB" ] || { echo "$JOBS/$JOB exists; a job is never overwritten" >&2; exit 1; }

mkdir -p "$JOBS" "$TRIALS"
chmod 755 "$TRIALS"

exec systemd-run --unit="taste-harbor-$JOB" --collect \
  --property=RuntimeMaxSec="${RUNTIME_MAX_SEC:-86400}" --property=WorkingDirectory="$JOBS" \
  --property=EnvironmentFile="$SECRETS" \
  --setenv=PYTHONPATH="$TASTE_SOURCE${EXTRA_PATH:+:$EXTRA_PATH}" \
  --setenv=HARBOR_TELEMETRY=0 --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=HOME=/root \
  --setenv=DOCKER_CONFIG="$DOCKER_CONFIG" --setenv=TASTE_TRIALS_ROOT="$TRIALS" \
  --setenv=PATH="$HARBOR_VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "$HARBOR_VENV/bin/harbor" run -p "$TASKS" \
    -a taste.benchmarks.harbor_agent:TasteAgent -m "$MODEL" \
    --ak worker_python="$WORKER_PYTHON" \
    -o "$JOBS" --job-name "$JOB" "$@"
