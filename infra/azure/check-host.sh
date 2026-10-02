#!/bin/sh
# Check that this host can run a Harbor job with the Taste agent.
#
#   sudo infra/azure/check-host.sh
#
# Read-only: it changes nothing and prints no secret. One line per check;
# exit status 1 if any check failed. The variables are the ones
# run-harbor.sh takes, with the same defaults.
set -u

TASTE_SOURCE=${TASTE_SOURCE:-$(cd "$(dirname "$0")/../.." && pwd)}
HARBOR_VENV=${HARBOR_VENV:-/root/taste-harbor-20260927/venv}
WORKER_PYTHON=${WORKER_PYTHON:-/home/bugbash/taste-openai-20260923.venv/bin/python}
WORKER_USER=${TASTE_WORKER_USER:-bugbash}
SECRETS=${SECRETS:-/root/taste-secrets/azure.env}
TRIALS=${TASTE_TRIALS_ROOT:-/var/lib/taste-trials}
DOCKER_CONFIG=${DOCKER_CONFIG:-/root/taste-harbor-20260927/docker-config}
export DOCKER_CONFIG

failed=0
ok()   { printf 'ok    %s\n' "$1"; }
warn() { printf 'warn  %s\n      %s\n' "$1" "$2"; }
fail() { printf 'FAIL  %s\n      %s\n' "$1" "$2"; failed=1; }
# check "<what>" "<how to fix>" command...
check() { what=$1; hint=$2; shift 2; if "$@" >/dev/null 2>&1; then ok "$what"; else fail "$what" "$hint"; fi; }

# --- the machine ---------------------------------------------------------------
check "running as root" "run this with sudo: the trial owner needs Docker and systemd" \
    test "$(id -u)" = 0
check "Linux with cgroup v2" "a unified cgroup hierarchy is required (/sys/fs/cgroup/cgroup.controllers)" \
    test -f /sys/fs/cgroup/cgroup.controllers
check "systemd-run is available" "install systemd; goals run in bounded transient units" \
    command -v systemd-run
check "Docker daemon answers" "start Docker, or check the socket at /var/run/docker.sock" \
    docker info
check "Docker Compose plugin" "install the Compose plugin (or set DOCKER_CONFIG to where it is)" \
    docker compose version
check "Docker Buildx plugin" "install the Buildx plugin: task images are built on first use" \
    docker buildx version
if command -v nft >/dev/null 2>&1; then ok "nftables is installed"
else warn "nftables is not installed" "tasks with a network allowlist need it for Harbor's egress control"; fi

# --- Harbor and this checkout --------------------------------------------------
check "Harbor is installed at $HARBOR_VENV" "create Harbor's environment there, or set HARBOR_VENV" \
    test -x "$HARBOR_VENV/bin/harbor"
check "checkout $TASTE_SOURCE is owned by root" "copy the checkout to a root-owned directory, or set TASTE_SOURCE" \
    test "$(stat -c %u "$TASTE_SOURCE" 2>/dev/null)" = 0
check "Harbor's Python can import the agent" "PYTHONPATH must reach this checkout, and Harbor must be importable" \
    env PYTHONPATH="$TASTE_SOURCE" "$HARBOR_VENV/bin/python" -c "import taste.benchmarks.harbor_agent"

# --- the unprivileged account that runs the goal ------------------------------
if id "$WORKER_USER" >/dev/null 2>&1; then
    ok "account $WORKER_USER exists"
    check "$WORKER_USER is not root" "the goal's processes need an unprivileged account" \
        test "$(id -u "$WORKER_USER")" != 0
    check "$WORKER_USER has no supplementary groups" "remove it from every extra group, above all docker" \
        test "$(id -G "$WORKER_USER" | wc -w)" = 1
    check "worker interpreter exists" "create the worker environment, or set WORKER_PYTHON" \
        test -x "$WORKER_PYTHON"
    check "$WORKER_USER can import taste, openai and git from this checkout" \
        "install '.[openai]' into the worker environment and make the checkout readable" \
        runuser -u "$WORKER_USER" -- "$WORKER_PYTHON" -I -c \
        "import sys; sys.path.insert(0, '$TASTE_SOURCE'); import taste, openai, git; assert sys.version_info >= (3, 11)"
    check "$WORKER_USER can run git" "install git" runuser -u "$WORKER_USER" -- git --version
else
    fail "account $WORKER_USER exists" "create it with no supplementary groups, or set TASTE_WORKER_USER"
fi

# --- credentials and storage ---------------------------------------------------
if [ -f "$SECRETS" ]; then
    check "$SECRETS is root-only (mode 600)" "chown root: and chmod 600 the file" \
        test "$(stat -c '%u %a' "$SECRETS")" = "0 600"
    for name in AZURE_OPENAI_BASE_URL AZURE_OPENAI_API_KEY; do
        check "$SECRETS sets $name" "add a line $name=... to the file" grep -q "^$name=." "$SECRETS"
    done
else
    fail "$SECRETS exists" "create it with AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_KEY, mode 600"
fi
if [ -d "$TRIALS" ]; then
    check "$TRIALS is writable by root alone" "chown root: and chmod 755 the directory" \
        test "$(stat -c '%u %a' "$TRIALS")" = "0 755"
else
    ok "$TRIALS will be created on the first run"
fi
free_kb=$(df -Pk /var/lib 2>/dev/null | awk 'NR==2 {print $4}')
if [ "${free_kb:-0}" -ge 10485760 ]; then ok "at least 10 GB free under /var/lib"
else warn "less than 10 GB free under /var/lib" "task images and trial records are kept there"; fi

[ "$failed" = 0 ] && echo "This host is ready." || echo "This host is not ready: fix the lines marked FAIL."
exit "$failed"
