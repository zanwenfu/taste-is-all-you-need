"""Ownership of trial resources survives setup, grading, and interrupt failures."""

from types import SimpleNamespace

import pytest

from taste.benchmarks import swebench_run as runner
from taste.benchmarks.swebench import SWEInstance
from taste.config import HarnessConfig
from taste.evalrun import Cell
from taste.memory import Memory
from taste.replay import SuiteProbe
from tests.test_evalrun import _run_result


@pytest.fixture
def resources(tiny_repo):
    closed = []
    sandbox = SimpleNamespace(workdir="/testbed", close=lambda: closed.append("sandbox"))
    instance = SWEInstance(instance_id="toy__toy-1", repo="psf/requests", base_commit="0" * 40,
                           problem_statement="synthetic", test_patch="", version="1",
                           fail_to_pass=(), pass_to_pass=(), image="synthetic")
    context = runner.CellContext(instance=instance, config=HarnessConfig.arm("A0"),
                                 workspace=tiny_repo, gitdir=tiny_repo / ".git" / "taste",
                                 session="synthetic", agent_sandbox=sandbox,
                                 provider=SimpleNamespace(open=lambda **kwargs: sandbox))
    return context, closed


@pytest.mark.parametrize("stage", ["materialize", "parity", "config"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_prepare_closes_container_when_ownership_cannot_transfer(tmp_path, monkeypatch, resources,
                                                                stage, exception):
    ctx, closed = resources

    def fail(*args, **kwargs):
        raise exception(stage)

    monkeypatch.setattr(runner.swebench, "materialize_from_image",
                        fail if stage == "materialize" else lambda *args: None)
    if stage == "config":
        monkeypatch.setattr(HarnessConfig, "arm", fail)
    prepare = runner.make_prepare(instances={ctx.instance.instance_id: ctx.instance}, root=tmp_path,
                                  provider=ctx.provider, route_execution=True,
                                  parity_check=fail if stage == "parity" else lambda *args: None)
    with pytest.raises(exception, match=stage):
        prepare(Cell(ctx.instance.instance_id, "A0", 1))
    assert closed == ["sandbox"]


@pytest.mark.parametrize("stage", ["llm_factory", "router", "kernel", "overrides", "run", "stats"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_execution_setup_and_receipt_failures_close_container(monkeypatch, resources, stage, exception):
    ctx, closed = resources

    def fail(*args, **kwargs):
        raise exception(stage)

    class LLM:
        @property
        def stats(self):
            if stage == "stats":
                fail()
            return SimpleNamespace(total_cost_usd=0.0, total_work_usd=0.0)

    monkeypatch.setattr("taste.routing.SandboxRouter",
                        fail if stage == "router" else lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "Kernel", fail if stage == "kernel" else lambda **kwargs:
                        SimpleNamespace(run=fail if stage == "run" else lambda **kwargs: _run_result()))
    execute = runner.make_execute(llm_factory=fail if stage == "llm_factory" else lambda ctx: LLM(),
                                  run_overrides=fail if stage == "overrides" else None)
    with pytest.raises(exception, match=stage):
        execute(Cell(ctx.instance.instance_id, "A0", 1), ctx)
    assert closed == ["sandbox"]
    assert ctx.agent_sandbox is None


@pytest.mark.parametrize("stage", ["timeline", "provider", "executor", "reconstruct", "grade", "evidence"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_scoring_releases_every_acquired_resource(monkeypatch, resources, tmp_path, stage, exception):
    ctx, closed = resources
    ctx.agent_sandbox = None
    original_close = Memory.close

    def close(memory):
        closed.append("memory")
        original_close(memory)

    def fail(*args, **kwargs):
        raise exception(stage)

    monkeypatch.setattr(Memory, "close", close)
    monkeypatch.setattr(runner, "load_timeline", fail if stage == "timeline" else lambda *args: [])
    if stage == "provider":
        ctx.provider.open = fail
    if stage == "executor":
        monkeypatch.setattr(runner, "SandboxProbeExecutor", fail)
    if stage == "reconstruct":
        monkeypatch.setattr(runner, "reconstruct", fail)
    if stage == "evidence":
        monkeypatch.setattr(runner.CellEvidence, "write", fail)
    score = runner.make_score(ledger_dir=tmp_path, grade=fail if stage == "grade" else None,
                              suite_factory=lambda instance: SuiteProbe("synthetic", "true", ()))
    with pytest.raises(exception, match=stage):
        score(Cell(ctx.instance.instance_id, "A0", 1), ctx, _run_result())
    assert closed == (["memory"] if stage in {"timeline", "provider"} else ["sandbox", "memory"])


def test_failed_sandbox_cleanup_still_releases_memory(monkeypatch, resources, tmp_path):
    ctx, closed = resources
    original_close = Memory.close

    def close_memory(memory):
        closed.append("memory")
        original_close(memory)

    def close_sandbox():
        closed.append("sandbox")
        raise RuntimeError("sandbox close failed")

    ctx.agent_sandbox.close = close_sandbox
    monkeypatch.setattr(Memory, "close", close_memory)
    monkeypatch.setattr(runner, "load_timeline", lambda *args: [])
    score = runner.make_score(ledger_dir=tmp_path,
                              suite_factory=lambda instance: SuiteProbe("synthetic", "true", ()))
    with pytest.raises(RuntimeError, match="sandbox close failed"):
        score(Cell(ctx.instance.instance_id, "A0", 1), ctx, _run_result())
    assert closed == ["sandbox", "memory"]
