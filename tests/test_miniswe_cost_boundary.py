"""Scaffold costs survive failures while the adapter publishes its manifest."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from taste.benchmarks import miniswe
from taste.evalrun import run_sweep


@pytest.mark.parametrize("agent_failed", [False, True])
@pytest.mark.parametrize("failure", ["manifest_write", "manifest_metadata"])
def test_manifest_failure_preserves_scaffold_spending(tiny_repo, tmp_path, monkeypatch,
                                                     agent_failed, failure):
    class Agent:
        def __init__(self, *args, **kwargs):
            self.cost = 0.0
            self.n_calls = 0

        def run(self, task):
            self.cost = 1.25  # Synthetic receipt; no model is contacted.
            self.n_calls = 1
            if agent_failed:
                raise RuntimeError("agent failed after paid call")
            return {"exit_status": "Submitted", "submission": "done"}

    for name in ("minisweagent", "minisweagent.agents", "minisweagent.agents.default"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["minisweagent.agents.default"], "DefaultAgent", Agent, raising=False)
    monkeypatch.setattr(miniswe, "load_scaffold_config", lambda *args, **kwargs: {
        "agent": {}, "model": {"model_name": "synthetic"},
    })
    monkeypatch.setattr("taste.routing.prepare_container_tree", lambda *args, **kwargs: "HEAD")
    if failure == "manifest_write":
        write = Path.write_text

        def fail_manifest(path, *args, **kwargs):
            if path.name == "manifest.json":
                raise OSError("manifest write failed")
            return write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_manifest)
    else:
        def fail_metadata(*args, **kwargs):
            raise RuntimeError("manifest metadata failed")

        monkeypatch.setattr(miniswe.TasteEnvironment, "get_template_vars", fail_metadata)

    closed = []
    context = SimpleNamespace(
        workspace=tiny_repo, gitdir=tiny_repo / ".git" / "taste",
        instance=SimpleNamespace(problem_statement="synthetic work"),
        agent_sandbox=SimpleNamespace(workdir="/testbed", network_mode="none",
                                      close=lambda: closed.append(True)),
        llm_stats=None,
    )
    execute = miniswe.make_miniswe_execute(model_name="synthetic", model_factory=object)
    report = run_sweep(tasks=["task"], arms=["MSWE"], trials=1, ledger_dir=tmp_path / "ledger",
                       prepare=lambda cell: context, execute=execute)
    record = report.results[0]
    assert record.status == "error" and "manifest" in record.error
    assert record.billed_usd == 1.25 and record.work_usd == 1.25
    assert context.llm_stats.total_cost_usd == 1.25
    assert closed == [True] and context.agent_sandbox is None
