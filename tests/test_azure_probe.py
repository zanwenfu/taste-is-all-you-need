"""Bounded live-check script tested through SDK HTTP before spending money."""

from __future__ import annotations

import json

import pytest

from scripts.check_azure_openai import run_probe
from tests.test_azure_openai import BINDINGS, config, httpx
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_openai_responses import function_call, message, response

sdk_transport = _sdk_transport


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_three_call_probe_persists_intents_and_replays_the_tool_result(tmp_path, sdk_transport):
    output = tmp_path / "probe.jsonl"

    def handle(wire):
        current = events(output)[-1]
        assert current["calls"][-1]["status"] == "dispatched"
        body = json.loads(wire.content)
        assert body["max_output_tokens"] == 512
        deployment = body["model"]
        if deployment == "gpt-6-astra":
            items = [message("READY")]
        elif any(item.get("type") == "function_call_output" for item in body["input"]):
            assert body["input"][-1]["output"] == "server-smoke-42"
            items = [message("server-smoke-42")]
        else:
            items = [function_call("{}", name="read_probe_value")]
        model = next(b.model for b in BINDINGS if b.deployment == deployment)
        return httpx.Response(200, json=response(model=deployment, output=items),
                              headers={"x-ms-served-model": model})

    sent, _ = sdk_transport(handle)
    report = run_probe(config(), output)
    assert report["status"] == "passed"
    assert len(sent) == len(report["calls"]) == 3
    assert events(output)[-1] == report
    assert report["billed_usd"] > 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert "azure-test-only" not in output.read_text()
    with pytest.raises(FileExistsError):
        run_probe(config(), output)
    assert len(sent) == 3


@pytest.mark.parametrize("failure", ["lost_reply", "wrong_snapshot", "wrong_answer"])
def test_failed_probe_stops_after_one_call_and_cannot_resume(tmp_path, sdk_transport, failure):
    output = tmp_path / "failed.jsonl"

    def handle(wire):
        if failure == "lost_reply":
            raise httpx.ReadError("private-error-marker", request=wire)
        if failure == "wrong_snapshot":
            return httpx.Response(200, json=response(model="gpt-6-astra", output=[message("READY")]))
        return httpx.Response(200, json=response(model=BINDINGS[0].model, output=[message("WRONG")]))

    sent, _ = sdk_transport(handle)
    report = run_probe(config(), output)
    assert report["status"] == "failed"
    assert len(sent) == 1
    assert len(report["calls"]) == 1
    assert "private-error-marker" not in output.read_text()
    assert "azure-test-only" not in output.read_text()
    expected = "received" if failure == "wrong_answer" else "dispatched"
    assert report["calls"][0]["status"] == expected
    assert events(output)[-1] == report
    with pytest.raises(FileExistsError):
        run_probe(config(), output)
    assert len(sent) == 1
