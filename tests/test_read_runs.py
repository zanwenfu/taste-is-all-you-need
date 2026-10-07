"""The trajectory reader over a recovery study: each run the checker rejected, read once."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from tests.recovery_fakes import NOT_DONE, agent_steps, record

_SPEC = importlib.util.spec_from_file_location(
    "read_runs", Path(__file__).resolve().parents[1] / "scripts" / "read_runs.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def state(trials):
    record(trials, "tok-a", agent_steps(6, tests_pass_at=(3,)), task="Make parse('') return [].")
    record(trials, "tok-known", agent_steps(4))
    rejected = {"verdict": "not_done", **NOT_DONE, "confidence": 0.8}

    def run(token, status, reply):
        return {"status": status, "base": {"token": token}, "check": {"reply": reply}}

    return {"runs": {"run-a": run("tok-a", "rejected", rejected),
                     "run-accepted": run("tok-a", "accepted", {"verdict": "done"}),
                     "run-known": run("tok-known", "rejected", rejected),
                     "run-unrecorded": run("tok-none", "rejected", rejected),
                     "run-unchecked": run("tok-a", "pending", None)}}


class FakeLLM:
    """Answers with ``replies`` in turn; each call costs $0.01."""

    def __init__(self, *replies):
        self.replies, self.calls, self.spent = list(replies), [], 0.0

    def call(self, **request):
        self.calls.append(request)
        self.spent += 0.01
        return SimpleNamespace(summary_text=self.replies.pop(0))

    def spent_usd(self):
        return self.spent


def test_each_rejected_run_is_read_once_with_the_checkers_findings(tmp_path):
    trials = tmp_path / "trials"
    seen = []

    def ask(messages):
        seen.append(messages)
        return '{"step": 3, "reason": "It stopped testing after the edit.", "confidence": 0.7}'

    found = list(cli.read_runs(state(trials), trials, ask, known={"run-known"}))
    assert [(run, why) for run, _, why in found] == [
        ("run-a", None), ("run-unrecorded", "no settled record of the agent's steps")]
    assert found[0][1] == {"step": 3, "reason": "It stopped testing after the edit.", "confidence": 0.7}
    # The reader saw the task, the steps and what the checker found.
    [messages] = seen
    prompt = messages[-1]["content"]
    assert "Make parse('') return []." in prompt and "Empty input must parse to an empty list" in prompt
    assert "python -m pytest" in prompt


def test_the_command_line_writes_readings_the_driver_reads_and_keeps_earlier_ones(tmp_path, capsys):
    trials, out, path = tmp_path / "trials", tmp_path / "readings.json", tmp_path / "rec.json"
    path.write_text(json.dumps(state(trials)))
    out.write_text(json.dumps({"run-known": {"step": 2, "reason": "earlier", "confidence": 0.5}}))
    llm = FakeLLM("not json", '{"step": 4, "reason": "Wrong file.", "confidence": 0.6}')
    assert cli.main(["--state", str(path), "--trials", str(trials), "--out", str(out)], llm=llm) == 0
    written = json.loads(out.read_text())
    assert written["run-known"]["reason"] == "earlier"
    # A reply that is not a reading is answered once with what was wrong; both calls are its cost.
    assert written["run-a"] == {"step": 4, "reason": "Wrong file.", "confidence": 0.6, "model": "gpt-6-sol",
                                "effort": "medium", "usd": 0.02}
    assert [call["model"] for call in llm.calls] == ["gpt-6-sol-2026-09-22"] * 2
    assert llm.calls[0]["effort"] == "medium" and llm.calls[0]["role"] == "reader"
    assert "run-unrecorded: not read: no settled record" in capsys.readouterr().out
    from taste.recovery_study.recovery_driver import readings
    assert readings(out)["run-a"]["step"] == 4


def test_the_default_budget_holds_the_worst_case_a_call_reserves():
    # The LLM reserves a call's worst case (the model's full context window) before sending it.
    from taste.benchmarks.harbor_settings import SERVED_MODELS
    from taste.pricing import max_call_cost_usd
    worst = max_call_cost_usd(SERVED_MODELS["gpt-6-sol"], max_output_tokens=cli.MAX_OUTPUT_TOKENS, cap_on="billed")
    assert worst < cli.BUDGET_USD < 2 * worst + 1
