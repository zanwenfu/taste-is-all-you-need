"""HTTP-only fixtures injected into actual Azure goal and worker processes."""
from __future__ import annotations

import importlib.abc
import json
import os
import sys
import threading


def install(counter, mode):
    class NoClaude(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("claude_agent_sdk"):
                raise AssertionError("Azure Harbor process imported Claude SDK")
    sys.meta_path.insert(0, NoClaude())
    import httpx

    from taste.brains.monitor_judge import TERMINAL_JUDGEMENT_SCHEMA
    from taste.providers.azure_openai import AZURE_PLANNER_MODEL, AZURE_WORKER_MODEL
    from tests.test_azure_worker_runtime import accepted
    from tests.test_openai_responses import function_call, message, response
    from tests.test_responses_monitor import verdict

    counts = {}
    lock = threading.Lock()

    def handle(wire):
        request = json.loads(wire.content)
        assert str(wire.url) == "https://test-resource.openai.azure.com/openai/v1/responses"
        assert wire.headers["authorization"] == "Bearer azure-test-only"
        role = ("planner" if request["model"] == "gpt-6-astra" else "worker" if "tools" in request
                else "terminal" if TERMINAL_JUDGEMENT_SCHEMA in wire.content.decode() else "monitor")
        with lock:
            counts[role] = number = counts.get(role, 0) + 1
            fd = os.open(counter, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, (json.dumps({"role": role, "number": number, "pid": os.getpid()}) + "\n").encode())
                os.fsync(fd)
            finally:
                os.close(fd)
        if mode == "killed-worker" and role == "worker" and number == 2:
            os.kill(os.getpid(), 9)
        if mode == "lost-planner" and role == "planner":
            os.kill(os.getpid(), 9)
        if role == "planner":
            payload = json.loads(request["input"][0]["content"])
            result = payload["required_output_shape"]
            delivered = any(row["run"]["phase"] == "delivered" for row in payload["request"]["world"]["outcomes"])
            if delivered:
                result["assignments"] = []
            else:
                assert number == 1, "fixture must not reissue undelivered work"
                assignment = result["assignments"][0]
                assignment["assignment_id"] = "terminal-result"
                assignment["contract"].update(identity="terminal-worker",
                    task="Use terminal_exec to write correct to /tmp/agent-result, then write output.txt containing correct",
                    outputs=["output.txt"], success_criteria=["terminal result and evidence contain correct"])
                assignment["outputs"][0].update(artifact_id="terminal-evidence", path="output.txt")
            result.update(complete=delivered, completion_reason="certified terminal evidence delivered" if delivered else "",
                          rationale="one terminal action with a certified evidence artifact")
            if "final_reply" in result["metadata"]:
                result["metadata"]["final_reply"] = (
                    "Wrote correct to /tmp/agent-result and retained the terminal evidence." if delivered else "")
            result["assessment"] = [{"criterion_id": item["criterion_id"], "verdict": "met" if delivered else "not_met",
                "evidence": "delivered terminal evidence" if delivered else "no result yet"} for item in payload["standing_criteria"]]
            output = [message(json.dumps(result))]
        elif role == "worker":
            if number == 1:
                command = ("touch /tmp/owner-command-started; (sleep 30 &); sleep 30" if mode == "killed-owner"
                           else "printf correct > /tmp/agent-result")
                if mode.startswith("separate-"):
                    command += "; mkdir -p /tmp/evidence; printf '\\000\\377data' > /tmp/evidence/raw.bin"
                    if mode.startswith("separate-service-") or mode.startswith("separate-kill-service-"):
                        command += "; python3 -c \"import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:8765/submit', data=b'correct', timeout=2).read() == b'ok'\""
                    if mode == "separate-missing":
                        command += "; rm /tmp/agent-result"
                    elif mode == "separate-link":
                        command += "; ln -s /etc/passwd /tmp/evidence/outside"
                output = [function_call(json.dumps({"command": command, "cwd": "/tmp", "timeout_seconds": 3}),
                                        name="terminal_exec", call_id="terminal_call")]
            elif number == 2:
                output = [function_call(json.dumps({"artifact": "output.txt", "body": "correct", "executable": False}),
                                        name="write_artifact", call_id="artifact_call")]
            else:
                output = [message(json.dumps(accepted(request)))]
        else:
            output = [message(verdict(terminal=role == "terminal"))]
        return httpx.Response(200, json=response(model=AZURE_PLANNER_MODEL if role == "planner" else AZURE_WORKER_MODEL, output=output))

    original_client = httpx.Client

    class Client(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(handle))
    httpx.Client = Client


def install_goal(counter, mode):
    install(counter, mode)
    import taste.brains.azure_central_host as host

    factory = host.worker_command_factory

    def wrapped_factory(*args, **kwargs):
        command = factory(*args, **kwargs)

        def wrapped(spec):
            argv = list(command(spec))
            code = f"from tests.azure_harbor_wire import install; install({counter!r}, {mode!r}); "
            argv[3] = argv[3].replace("import runpy;", code + "import runpy;", 1)
            return argv
        return wrapped
    host.worker_command_factory = wrapped_factory
