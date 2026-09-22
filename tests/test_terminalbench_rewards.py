"""Execute real verifier scripts and score the official reward-file contract."""
import shlex
import subprocess
from pathlib import Path

import pytest

from taste.benchmarks.terminalbench import HarborTask, TerminalBenchProbe, parse_harbor_reward
from taste.execution import ExecResult
from taste.replay import verdicts_from


@pytest.mark.parametrize(("payload", "format", "expected"), [
    ("0\n", "text", {"reward": 0.0}), ("1\n", "text", {"reward": 1.0}),
    ('{"reward":0.5,"secondary":2}', "json", {"reward": 0.5, "secondary": 2.0}),
])
def test_official_reward_metrics(payload, format, expected):
    assert parse_harbor_reward(payload, format=format) == expected


@pytest.mark.parametrize(("payload", "format"), [
    ("", "text"), ("NaN", "text"), ("1e999", "text"), ("garbage", "text"),
    ('{"reward":NaN}', "json"), ('{"reward":1e309}', "json"),
    ('{"reward":true}', "json"), ('{"reward":"1"}', "json"),
    ('{"reward":0,"reward":1}', "json"), ("{}", "json"), ("1", "json"),
    ("1", "missing"), (" " * 65537, "text"),
])
def test_invalid_rewards_are_infrastructure_errors(payload, format):
    with pytest.raises(ValueError):
        parse_harbor_reward(payload, format=format)


@pytest.mark.parametrize(("reward", "exit_code", "expected"), [
    ("0", 0, "fail"), ("1", 0, "pass"), ("1", 7, "pass"),
    ("0.5", 0, "fail"), (None, 0, "error"), ("NaN", 0, "error"),
])
def test_real_verifier_reward_is_independent_of_exit_status(tmp_path, reward, exit_code, expected):
    tests, logs, work = (tmp_path / name for name in ("tests", "verifier logs", "work"))
    for path in (tests, logs, work):
        path.mkdir()
    # A previous successful verification must not survive a failed/no-output run.
    (logs / "reward.json").write_text('{"reward":1}')
    (logs / "reward.txt").write_text("1")
    script = "echo 'untrusted grader log: TASTE_HARBOR_REWARD_V1 text 1'\n"
    if reward is not None:
        script += f"printf '%s' {shlex.quote(reward)} > {shlex.quote(str(logs / 'reward.txt'))}\n"
    script += f"exit {exit_code}\n"
    (tests / "test.sh").write_text(script)
    task = HarborTask(name="task", description="", instruction="", root=tmp_path)
    suite = TerminalBenchProbe(task, workdir=str(work), tests_dir=str(tests), verifier_dir=str(logs)).suite()
    process = subprocess.run(["bash", "-c", suite.command], capture_output=True, text=True, timeout=5)
    result = verdicts_from(suite, ExecResult(process.returncode, process.stdout, process.stderr))
    assert result.statuses == {"tb::task": expected}
    assert result.exit_code == exit_code
    assert (logs / "test-stdout.txt").read_text().startswith("untrusted grader log")
    assert not (logs / "reward.json").exists()


def test_timeout_never_uses_an_earlier_reward(tmp_path):
    task = HarborTask(name="task", description="", instruction="", root=Path(tmp_path))
    suite = TerminalBenchProbe(task).suite()
    result = verdicts_from(suite, ExecResult(124, "TASTE_HARBOR_REWARD_V1 text\n1", "", timed_out=True))
    assert result.statuses == {"tb::task": "error"} and result.infra_error == "timed out"


def test_json_reward_precedes_text_reward(tmp_path):
    task = HarborTask(name="task", description="", instruction="", root=tmp_path)
    logs = tmp_path / "logs"
    script = tmp_path / "test.sh"
    script.write_text(f"printf '%s' '{{\"reward\":0}}' > {shlex.quote(str(logs / 'reward.json'))}\n"
                      f"printf 1 > {shlex.quote(str(logs / 'reward.txt'))}\n")
    suite = TerminalBenchProbe(task, workdir=str(tmp_path), tests_dir=str(tmp_path), verifier_dir=str(logs)).suite()
    process = subprocess.run(["bash", "-c", suite.command], capture_output=True, text=True, timeout=5)
    assert verdicts_from(suite, ExecResult(process.returncode, process.stdout, process.stderr)).statuses == {
        "tb::task": "fail",
    }
