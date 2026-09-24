"""Bounded, serial Azure-only validation. Run on the authorized Linux server.

Reads only Azure endpoint/key JSON from stdin; never persists credentials.
At most three paid calls, each capped at 512 output tokens, no retries. This
checks provider/tool replay, not the Claude-free worker or a benchmark task.
An existing report is never reused: interrupted dispatches remain unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taste.llm import LLM
from taste.pricing import table_sha
from taste.providers.azure_openai import (
    AZURE_PLANNER_MODEL,
    AZURE_WORKER_MODEL,
    AzureDeployment,
    AzureOpenAIConfig,
)

_TOOL = {
    "name": "read_probe_value", "description": "Read the fixed validation value.",
    "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
}
_VALUE = "server-smoke-42"


def _persist(path: Path, report: dict[str, Any], *, create: bool = False) -> None:
    # Private, append-only event log. Each event contains the full report so a
    # crash leaves earlier valid lines and a durable dispatch intent to audit.
    flags = os.O_WRONLY | os.O_APPEND | (os.O_CREAT | os.O_EXCL if create else 0)
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    if create:
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)


def run_probe(config: AzureOpenAIConfig, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    source_files = [
        "scripts/check_azure_openai.py", "taste/llm.py", "taste/pricing.py",
        "taste/providers/base.py", "taste/providers/_openai.py", "taste/providers/azure_openai.py",
    ]
    report: dict[str, Any] = {
        "schema": "taste.azure-openai-probe.v1", "status": "initialized", "calls": [],
        "endpoint": config.base_url, "max_calls": 3, "max_output_tokens_per_call": 512,
        "pricing_sha": table_sha(), "sdk_version": importlib.metadata.version("openai"),
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_files},
    }
    _persist(output, report, create=True)
    # The facade reserves a full context window before each call, even though
    # these fixed prompts are tiny. The physical request bounds are three
    # constant prompts/tool exchanges and at most 1,536 generated tokens.
    llm = LLM(azure_openai=config, budget_usd=100, cap_on="billed", max_attempts=1)

    def call(stage: str, model: str, **kwargs: Any):
        if len(report["calls"]) >= 3:
            raise RuntimeError("probe call limit exceeded")
        row = {"stage": stage, "model": model, "status": "dispatched"}
        report["calls"].append(row)
        report["status"] = "running"
        _persist(output, report)
        result = llm.call(model=model, max_tokens=512, effort="low", temperature=None,
                          timeout_seconds=60, role=stage, **kwargs)
        row.update({
            "status": "received", "stop_reason": result.stop_reason,
            "usage": asdict(result.usage), "provenance": dict(result.provenance),
            "text": result.summary_text, "transcript": list(result.transcript_blocks),
        })
        report["billed_usd"] = llm.stats.total_cost_usd
        report["work_usd"] = llm.stats.total_work_usd
        _persist(output, report)
        return result

    try:
        llm.ensure_ready(AZURE_PLANNER_MODEL, AZURE_WORKER_MODEL)
        planner = call("planner", AZURE_PLANNER_MODEL, system="Reply exactly READY.", messages=[{"role": "user", "content": "Check readiness."}])
        if planner.stop_reason != "end_turn" or planner.summary_text != "READY":
            raise RuntimeError("Astra response contract failed")
        system = "Call read_probe_value exactly once. After its result, reply with that exact value only."
        messages = [{"role": "user", "content": "Read the validation value."}]
        worker = call("worker_tool", AZURE_WORKER_MODEL, system=system, messages=messages, tools=[_TOOL])
        if worker.stop_reason != "tool_use" or len(worker.tool_calls) != 1:
            raise RuntimeError("Sol tool-call contract failed")
        tool = worker.tool_calls[0]
        if tool.name != "read_probe_value" or tool.arguments != {}:
            raise RuntimeError("Sol requested an unexpected tool or arguments")
        # Use a JSON round trip, as durable memory does, before replaying the
        # opaque reasoning/message/tool items. No host or terminal tool runs.
        messages.extend([
            {"role": "assistant", "content": json.loads(json.dumps(worker.transcript_blocks))},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool.id, "content": _VALUE}]},
        ])
        final = call("worker_result", AZURE_WORKER_MODEL, system=system, messages=messages, tools=[_TOOL])
        if final.stop_reason != "end_turn" or final.summary_text != _VALUE:
            raise RuntimeError("Sol tool-result replay failed")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__  # SDK error text may contain sensitive headers.
    finally:
        for provider in llm._providers.values():
            if provider._client is not None:
                try:
                    provider._client.close()
                except BaseException as exc:
                    report["status"] = "failed"
                    report["close_error_type"] = type(exc).__name__
        _persist(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run this paid validation on the authorized Linux server")
    raw = sys.stdin.buffer.read(8193)
    if len(raw) > 8192:
        parser.error("credential input exceeds its limit")
    credentials = json.loads(raw)
    if not isinstance(credentials, dict) or set(credentials) != {"AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL"}:
        parser.error("stdin must contain only the two Azure credential fields")
    config = AzureOpenAIConfig.from_environment(credentials, deployments=(
        AzureDeployment(AZURE_PLANNER_MODEL, "gpt-6-astra"),
        AzureDeployment(AZURE_WORKER_MODEL, "gpt-6-sol"),
    ))
    result = run_probe(config, args.output)
    print(json.dumps({k: result.get(k) for k in ("status", "error_type", "billed_usd", "work_usd")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
