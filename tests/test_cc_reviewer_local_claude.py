"""The Claude Code reviewer with the real ``claude`` binary, a scripted local model and no network (opt-in).

    TASTE_CC_LOCAL_CLAUDE=1 nice -n 10 python -m pytest tests/test_cc_reviewer_local_claude.py

The binary is pointed at a local stand-in for the Messages API (``tests.cc_mock_api``) with a dummy key that is not a
key, and on macOS runs under ``sandbox-exec`` with outbound network allowed only to this machine; the subscription's
token is never read. The scripted "model" sends the commands an escaping reviewer would, and the tests read what the
binary did with them: every command reached only the docker stand-in (which runs nothing), as one ``docker exec``
into the named container; the host's shell expanded nothing; the refused and last-step messages are the checker's.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from taste.agents.checker import REFUSED, checker_task
from taste.recovery_study.cc_reviewer import (
    LAST_STEP,
    NO_STEPS,
    ReviewSettings,
    Runner,
    StopJob,
    reviewer_system_prompt,
)
from tests import cc_docker_shim

REAL = shutil.which("claude")
pytestmark = pytest.mark.skipif(os.environ.get("TASTE_CC_LOCAL_CLAUDE") != "1" or REAL is None,
                                reason="opt-in: TASTE_CC_LOCAL_CLAUDE=1 and claude on PATH")
TASK = checker_task("Make add() in calc.py return the sum of its two arguments.", "Fixed.", "")
DONE = {"verdict": "done", "confidence": 0.9}


@pytest.fixture
def local(tmp_path):
    bin_dir, toy = tmp_path / "bin", tmp_path / "toy"
    bin_dir.mkdir()
    toy.mkdir()
    (toy / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    cc_docker_shim.install(bin_dir, toy, tmp_path / "docker.jsonl")
    profile = tmp_path / "net.sb"
    profile.write_text('(version 1)\n(allow default)\n(deny network-outbound)\n'
                       '(allow network-outbound (remote ip "localhost:*"))\n'
                       '(allow network-outbound (remote unix-socket))\n')
    servers = []

    def start(script):
        (tmp_path / "script.json").write_text(json.dumps(script))
        port_file = tmp_path / "port"
        server = subprocess.Popen([sys.executable, "-I", str(Path(__file__).with_name("cc_mock_api.py")),
                                   "--script", str(tmp_path / "script.json"), "--log", str(tmp_path / "api.jsonl"),
                                   "--port-file", str(port_file)])
        servers.append(server)
        for _ in range(200):
            if port_file.exists() and port_file.read_text():
                break
            time.sleep(0.05)
        sandbox = f"sandbox-exec -f {shlex.quote(str(profile))} " if shutil.which("sandbox-exec") else ""
        wrapper = bin_dir / "claude-local"
        wrapper.write_text("#!/bin/sh\nunset CLAUDE_CODE_OAUTH_TOKEN\n"
                           f"export ANTHROPIC_BASE_URL=http://127.0.0.1:{port_file.read_text()}\n"
                           "export ANTHROPIC_API_KEY=mock-not-a-key\n"
                           f"exec {sandbox}{shlex.quote(REAL)} \"$@\"\n")
        wrapper.chmod(0o755)
        return wrapper

    class Local:
        def review(self, script, **settings):
            wrapper = start(script)
            runner = Runner("dummy-token-for-a-local-test", ReviewSettings(claude=str(wrapper), **settings),
                            environ={"PATH": f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "en_US.UTF-8"})
            return runner.review("c0ffee", "/app", TASK, tmp_path / "work")

        def requests(self):
            lines = (tmp_path / "api.jsonl").read_text().splitlines()
            return [json.loads(line) for line in lines if "_path" not in line]

        def docker(self):
            path = tmp_path / "docker.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    yield Local()
    for server in servers:
        server.terminate()


def bash(command):
    return {"type": "tool_use", "name": "Bash", "input": {"command": command, "description": "check"}}


def verdict(value):
    return [{"type": "tool_use", "name": "StructuredOutput", "input": value}]


def tool_results(request):
    out = []
    for message in request["messages"]:
        for block in message["content"] if isinstance(message["content"], list) else ():
            if block.get("type") == "tool_result":
                content = block["content"]
                out.append(content if isinstance(content, str) else json.dumps(content))
    return out


def test_every_command_runs_only_as_docker_exec_into_the_container(local):
    user = os.environ.get("USER") or "nobody-here"
    review = local.review([[bash("ls")], [bash("cat calc.py ; echo HOSTSIDE")], [bash("echo $(whoami) ~")],
                           [bash("cat /tests/test_calc.py")], verdict(DONE)])
    assert review.reply["verdict"] == "done" and review.reply["ended"] == "verdict"
    assert (review.reply["commands"], review.reply["refused"], review.reply["steps"]) == (3, 1, 5)
    assert review.hook["token_in_env"] is False
    commands = [call["command"] for call in local.docker() if "command" in call]
    assert commands == ["ls", "cat calc.py ; echo HOSTSIDE", "echo $(whoami) ~"]
    # Every docker call names the review's own container (the transport check's inspect calls name it as "inspect").
    assert {call.get("container") or call.get("inspect") for call in local.docker()} == {"c0ffee"}
    results = tool_results(local.requests()[-1])
    assert results[0].rstrip() == "exit code: 0\noutput:\ncalc.py"            # the container's folder, not the host's
    assert "HOSTSIDE" not in results[1].replace("cat: HOSTSIDE", "")
    assert user not in results[2] and "exit code: 127" in results[2]   # nothing was expanded by the host's shell
    assert REFUSED.format(name="/tests") in results[3]
    first = local.requests()[0]
    assert reviewer_system_prompt("/app", ReviewSettings()) in [block["text"] for block in first["system"]]
    assert {tool["name"] for tool in first["tools"]} <= {"Bash", "StructuredOutput", "DeferredToolPlaceholder"}
    assert first["max_tokens"] == 16384


def test_the_last_step_is_announced_before_the_last_call(local):
    review = local.review([[bash("ls")], [bash("ls")], [bash("ls")], [bash("ls")]], max_steps=3)
    requests = local.requests()
    assert len(requests) == 3
    assert LAST_STEP in json.dumps(requests[2]["messages"][-3:]) and LAST_STEP not in json.dumps(requests[1])
    assert review.reply["verdict"] is None and review.reply["ended"] == "steps"
    assert review.reply["commands"] == 2 and review.hook["denied"] == 1
    assert [event["kind"] for event in review.hook["events"]][-3:] == ["last_step", "no_steps", "ended_at_limit"]
    assert NO_STEPS in tool_results(requests[2])[-1] or len(requests) == 3


def test_a_verdict_the_checker_rejects_is_corrected_in_the_same_session(local):
    fenced = {"verdict": "not_done", "confidence": 0.7, "unmet_requirement": "```\nreturn a + b\n```",
              "evidence": {"check": "c", "command": "cat calc.py", "output": "", "expected": "e", "observed": "o"}}
    fixed = {**fenced, "unmet_requirement": "add() must return the sum."}
    review = local.review([verdict(fenced), [{"type": "text", "text": "ok"}], verdict(fixed)])
    assert review.reply["verdict"] == "not_done" and review.claude_code["corrections"] == 1
    assert "Your verdict was not accepted" in json.dumps(local.requests()[-1]["messages"])


def test_a_hook_that_cannot_run_blocks_every_command_and_stops_the_job(local):
    with pytest.raises(StopJob) as caught:
        local.review([[bash("ls")], verdict(DONE)], python="/nonexistent/python3")
    assert caught.value.kind == "hook_bypassed"
    assert [call for call in local.docker() if "command" in call] == []
    assert "hook error" in json.dumps(local.requests()[-1]["messages"])


def test_a_script_claude_code_cannot_read_still_runs_in_the_container(local):
    # Sun Nov 15: Claude Code refused heredoc scripts with rm in -p ("requires approval, ... no approval surface"),
    # even in bypass mode, so 10-18% of reviews failed closed. A reviewer's script must reach the container.
    # A live reviewer's command (vuls, Nov 15), denied without the PermissionRequest hook even in bypass mode.
    script = ("cd /app && cat > reporter/sbom/zz_eval_test.go <<'EOF'\npackage sbom\nimport \"testing\"\n"
              "func TestEvalParse(t *testing.T) {\n\tcases := [][3]string{\n"
              "\t\t{\"npm\",\"@babel/core\",\"@babel|core|\"},\n\t}\n\tfor _, c := range cases {\n"
              "\t\tns, n, sp := parsePkgName(c[0], c[1])\n\t\tgot := ns+\"|\"+n+\"|\"+sp\n"
              "\t\tif got != c[2] { t.Errorf(\"%s %s: got %q want %q\", c[0], c[1], got, c[2]) }\n\t}\n}\nEOF\n"
              "timeout 110 go test ./reporter/sbom/ -run TestEvalParse -v 2>&1 | tail -20; echo done")
    review = local.review([[bash(script)], verdict(DONE)])
    results = tool_results(local.requests()[-1])
    assert not any(r.startswith("Permission for this tool use was denied") for r in results), results
    assert [call["command"] for call in local.docker() if "command" in call] == [script]
    assert [event["kind"] for event in review.hook["events"]] == ["command", "approved"]
