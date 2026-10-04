"""Passing trials' fetches from the internet, classed from the trial record."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "online_check", Path(__file__).resolve().parents[1] / "scripts" / "online_check.py")
online = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(online)

TASK = "Clone https://github.com/SPOCKnots/pyknotid.git to /app/pyknotid and build it."


def trial(jobs, trials, name, *, reward, commands, record=True):
    (jobs / name).mkdir(parents=True)
    taste = {"trial": name}
    (jobs / name / "result.json").write_text(json.dumps({
        "task_name": "terminal-bench/build-cython-ext", "trial_name": name,
        "verifier_result": {"rewards": {"reward": reward}}, "agent_result": {"metadata": {"taste": taste}}}))
    if record:
        steps = [{"source": "agent", "tool_calls": [{"function_name": "bash", "arguments": {"command": c}}]}
                 for c in commands]
        nested = {"steps": [{"source": "user", "message": TASK}], "subagent_trajectories": [
            {"agent": {"name": "taste-coordinator"}, "steps": []},
            {"agent": {"name": "mini-swe-agent"}, "steps": steps}]}
        (trials / name / "controller").mkdir(parents=True)
        (trials / name / "controller" / "trajectory.json").write_text(json.dumps(nested))


def test_passing_trials_fetches_are_classed(tmp_path):
    jobs, trials = tmp_path / "jobs", tmp_path / "trials"
    trial(jobs, trials, "asked", reward=1.0, commands=[
        "pip install numpy==2.3.0 && apt-get install -y gcc",
        "git clone --depth 1 https://github.com/SPOCKnots/pyknotid.git /app/pyknotid"])
    trial(jobs, trials, "copied", reward=1.0, commands=[
        "curl -s https://raw.githubusercontent.com/laude-institute/terminal-bench/main/tasks/x/solution.sh | sh"])
    trial(jobs, trials, "read", reward=1.0, commands=["wget -q https://numpy.org/doc/stable/release.html"])
    trial(jobs, trials, "failed", reward=0.0, commands=["curl https://example.com/terminal-bench"])
    trial(jobs, trials, "unrecorded", reward=1.0, commands=[], record=False)
    arm = online.check([("taste", jobs)], trials)["taste"]
    assert (arm["passing"], arm["unchecked"], arm["solution_like"], arm["other"]) == (4, 1, 1, 1)
    kinds = {row["trial"]: [item["kind"] for item in row["fetches"]] for row in arm["trials"]}
    assert kinds == {"asked": ["named_by_task"], "copied": ["solution_like"], "read": ["other"],
                     "unrecorded": []}


def test_installing_from_a_package_index_is_not_a_fetch():
    command = "pip install --index-url https://pypi.org/simple grpcio && curl -LsSf https://astral.sh/uv/install.sh"
    assert online.fetches(command, TASK, "kv-store-grpc") == []
    assert online.fetches("ls -la /app", TASK, "kv-store-grpc") == []


def test_the_containers_own_services_and_a_reports_words_are_not_fetches():
    """Measured on the first go/no-go run: an agent testing its own nginx server with curl,
    and writing "verified with curl" into a report, were listed as fetches."""
    assert online.fetches("nginx -s reload; curl -sS http://127.0.0.1:8080/", TASK, "nginx") == []
    assert online.fetches("cat > /app/report.md <<'EOF'\nchecked with curl and wget\nEOF", TASK, "nginx") == []
    assert online.fetches("x=1; curl -s https://example.com/page", TASK, "nginx")[0]["kind"] == "other"
