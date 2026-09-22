"""Unconfirmed Docker cleanup cannot disappear across adapters or sweep restart.

These are synthetic Docker API tests, not proof of real daemon/process cleanup.
"""

import json
from types import SimpleNamespace

import pytest

from taste.benchmarks import miniswe
from taste.benchmarks import swebench_run as runner
from taste.config import HarnessConfig
from taste.evalrun import Cell, CellResult, GradingResult, Ledger, resume_grading, run_sweep
from taste.execution import DockerProvider, DockerSandbox
from taste.ledger_costs import ledger_billed_usd
from taste.sweep_journal import UnsettledSweepAttempt
from tests.test_evalrun import _run_result
from tests.test_execution import _FakeContainer, _respawning_client


@pytest.fixture
def owned(monkeypatch):
    client = _respawning_client()
    provider = DockerProvider(client=client)
    sandbox = provider.open(key="first", image="synthetic")
    original_remove = sandbox.container.remove

    def unavailable(*args, **kwargs):
        raise ConnectionError("synthetic Docker removal failure")

    monkeypatch.setattr(sandbox.container, "remove", unavailable)
    return provider, sandbox, client, original_remove


def test_failed_removal_retains_ownership_and_prevents_reuse_until_explicit_cleanup(owned):
    provider, sandbox, client, remove = owned
    with pytest.raises(RuntimeError, match="removal failure"):
        sandbox.close()
    assert provider._open["first"] is sandbox
    assert not sandbox.container.removed
    with pytest.raises(RuntimeError, match="cleanup"):
        provider.open(key="first", image="synthetic")
    assert len(client.containers.spawned) == 1
    sandbox.container.remove = remove
    sandbox.close()
    replacement = provider.open(key="first", image="synthetic")
    assert replacement is not sandbox and sandbox.container.removed


def test_failed_cache_callback_can_retry_without_removing_the_container_twice():
    client = _respawning_client()
    calls = []

    def evict(sandbox):
        calls.append(sandbox.container.id)
        if len(calls) == 1:
            raise RuntimeError("cache callback failed")

    sandbox = DockerSandbox(image="synthetic", name="standalone", client=client, on_close=evict)
    removes = []
    remove = sandbox.container.remove

    def counted(**kwargs):
        removes.append(True)
        remove(**kwargs)

    sandbox.container.remove = counted
    with pytest.raises(RuntimeError, match="cache callback failed"):
        sandbox.close()
    assert sandbox.container.removed
    sandbox.close()
    sandbox.close()
    assert len(calls) == 2 and removes == [True]


def test_explicit_container_not_found_is_an_idempotent_close(monkeypatch):
    class NotFound(Exception):
        status_code = 404

    client = _respawning_client()
    provider = DockerProvider(client=client)
    sandbox = provider.open(key="first", image="synthetic")

    def already_gone(**kwargs):
        raise NotFound("no such container")

    monkeypatch.setattr(sandbox.container, "remove", already_gone)
    sandbox.close()
    sandbox.close()
    assert not provider._open


@pytest.mark.parametrize("failure", [ConnectionError, PermissionError])
def test_daemon_inspection_failure_does_not_evict_remove_or_replace(monkeypatch, failure):
    client = _respawning_client()
    provider = DockerProvider(client=client)
    sandbox = provider.open(key="first", image="synthetic")

    def unavailable(ref):
        raise failure("inspection unavailable")

    monkeypatch.setattr(client.containers, "get", unavailable)
    with pytest.raises(RuntimeError, match="inspection unavailable"):
        provider.open(key="first", image="synthetic")
    assert provider._open["first"] is sandbox
    assert len(client.containers.spawned) == 1 and not sandbox.container.removed


@pytest.mark.parametrize("status", ["paused", "restarting", "created", None, "malformed"])
def test_nonterminal_or_unknown_container_state_is_not_permission_to_remove(status):
    client = _respawning_client()
    provider = DockerProvider(client=client)
    sandbox = provider.open(key="first", image="synthetic")
    sandbox.container.status = status
    with pytest.raises(RuntimeError, match="state"):
        provider.open(key="first", image="synthetic")
    assert not sandbox.container.removed and len(client.containers.spawned) == 1


def test_close_all_attempts_every_resource_but_retains_the_failed_owner(owned):
    provider, first, _client, remove = owned
    second = provider.open(key="second", image="synthetic")
    with pytest.raises(RuntimeError, match="removal failure"):
        provider.close_all()
    assert second.container.removed and not first.container.removed
    assert list(provider._open) == ["first"]
    first.container.remove = remove
    provider.close_all()
    assert first.container.removed and not provider._open


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_exception_from_workdir_probe_closes_an_already_created_container(monkeypatch, failure):
    client = _respawning_client()

    def fail_probe(*args, **kwargs):
        raise failure("workdir probe failed")

    monkeypatch.setattr(_FakeContainer, "exec_run", fail_probe)
    with pytest.raises(failure, match="workdir probe failed"):
        DockerSandbox(image="synthetic", name="probe", client=client)
    assert len(client.containers.spawned) == 1
    assert client.containers.spawned[0].removed


def test_lost_container_creation_reply_leaves_owned_name_and_blocks_new_preparation(tmp_path, monkeypatch):
    client = _respawning_client()
    create = client.containers.run

    def lose_reply(image, **kwargs):
        create(image, **kwargs)
        raise ConnectionError("create reply lost")

    monkeypatch.setattr(client.containers, "run", lose_reply)
    provider = DockerProvider(client=client)
    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        run_sweep(tasks=["first", "must-not-start"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: provider.open(key=cell.key, image="synthetic"),
                  execute=lambda cell, ctx: pytest.fail("no execution"))
    assert len(client.containers.spawned) == 1
    _directory, receipt = _resources_receipt(tmp_path)
    assert receipt["resources"][0]["resource_id"] == client.containers.spawned[0].name
    assert receipt["billed_usd"] == 0.0
    _assert_fenced(tmp_path)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_miniswe_configuration_failure_is_inside_container_ownership(monkeypatch, failure):
    closed = []
    ctx = SimpleNamespace(agent_sandbox=SimpleNamespace(close=lambda: closed.append(True)))

    def fail_config(*args, **kwargs):
        raise failure("configuration failed")

    monkeypatch.setattr(miniswe, "load_scaffold_config", fail_config)
    execute = miniswe.make_miniswe_execute(model_name="synthetic")
    with pytest.raises(failure, match="configuration failed"):
        execute(Cell("task", "MSWE", 1), ctx)
    assert closed == [True] and ctx.agent_sandbox is None


def _resources_receipt(ledger):
    admission = json.loads((ledger / ".sweep-journal/pending.json").read_text())
    directory = ledger / ".sweep-journal" / admission["attempt_id"]
    return directory, json.loads((directory / "resources.json").read_text())["record"]


def _assert_fenced(ledger):
    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        run_sweep(tasks=["other"], arms=["B"], trials=1, retry_budget=100, ledger_dir=ledger,
                  prepare=lambda cell: pytest.fail("no new preparation"),
                  execute=lambda cell, ctx: pytest.fail("no paid retry"))
    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        resume_grading(ledger_dir=ledger, score=lambda record: pytest.fail("no recovery grading"))
    with pytest.raises(ValueError, match="pending"):
        ledger_billed_usd(ledger)


@pytest.mark.parametrize("phase", ["prepare", "execute", "score"])
@pytest.mark.parametrize("known_cost", [False, True])
def test_cleanup_fences_sweep_and_recovery_independently_of_known_spending(
        tmp_path, owned, phase, known_cost):
    _provider, sandbox, _client, _remove = owned
    started = []
    stats = SimpleNamespace(total_cost_usd=1.25, total_work_usd=2.0, cache_delta_usd=0.0)
    ctx = SimpleNamespace(llm_stats=stats if known_cost else None)

    def prepare(cell):
        started.append(cell.task)
        if phase == "prepare":
            sandbox.close()
        return ctx

    def execute(cell, context):
        if phase == "execute":
            sandbox.close()
        result = _run_result()
        result.stats = stats
        return result

    def score(cell, context, result):
        # A grader cannot change which execution receipt the resource failure
        # carries, even if it replaces context statistics before failing.
        context.llm_stats = None
        sandbox.close()
        return 1.0

    ledger = tmp_path / "ledger"
    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        run_sweep(tasks=["first", "must-not-start"], arms=["A"], trials=1, ledger_dir=ledger,
                  prepare=prepare, execute=execute, score=score)
    assert started == ["first"]
    assert Ledger(ledger).all_results() == []
    directory, receipt = _resources_receipt(ledger)
    expected = (0.0, 0.0) if phase == "prepare" else (
        (1.25, 2.0) if known_cost or phase == "score" else (None, None)
    )
    assert (receipt["billed_usd"], receipt["work_usd"]) == expected
    assert receipt["phase"] == phase
    assert receipt["resources"][0]["resource_id"] == sandbox.container.id
    assert not (directory / "ending.json").exists()
    _assert_fenced(ledger)


def test_miniswe_paid_result_does_not_hide_cleanup_failure_or_release_handle(tmp_path, owned, monkeypatch):
    _provider, sandbox, _client, _remove = owned
    ctx = SimpleNamespace(agent_sandbox=sandbox, llm_stats=None)
    monkeypatch.setattr(miniswe, "load_scaffold_config", lambda *args, **kwargs: {})

    def paid(context, *args):
        context.llm_stats = SimpleNamespace(total_cost_usd=1.25, total_work_usd=1.25)
        result = _run_result()
        result.stats = context.llm_stats
        return result

    monkeypatch.setattr(miniswe, "_run_cell", paid)
    ledger = tmp_path / "ledger"
    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        run_sweep(tasks=["first", "second"], arms=["MSWE"], trials=1, ledger_dir=ledger,
                  prepare=lambda cell: ctx, execute=miniswe.make_miniswe_execute(model_name="synthetic"))
    assert ctx.agent_sandbox is sandbox and not sandbox.container.removed
    _directory, receipt = _resources_receipt(ledger)
    assert receipt["billed_usd"] == 1.25
    _assert_fenced(ledger)


def test_swe_adapter_paid_result_retains_cost_and_container_until_cleanup(tmp_path, owned, monkeypatch):
    _provider, sandbox, _client, _remove = owned
    stats = SimpleNamespace(total_cost_usd=1.25, total_work_usd=2.0)
    ctx = runner.CellContext(
        instance=SimpleNamespace(instance_id="first", repo="synthetic", problem_statement="task"),
        config=HarnessConfig.arm("A0"), workspace=tmp_path, gitdir=tmp_path / ".git/taste",
        session="synthetic", agent_sandbox=sandbox,
    )
    monkeypatch.setattr("taste.routing.SandboxRouter", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "Kernel", lambda **kwargs: SimpleNamespace(run=lambda **kwargs: _run_result()))
    execute = runner.make_execute(llm_factory=lambda context: SimpleNamespace(stats=stats))
    ledger = tmp_path / "ledger"
    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        run_sweep(tasks=["first", "second"], arms=["A0"], trials=1, ledger_dir=ledger,
                  prepare=lambda cell: ctx, execute=execute)
    assert ctx.agent_sandbox is sandbox and not sandbox.container.removed
    _directory, receipt = _resources_receipt(ledger)
    assert (receipt["billed_usd"], receipt["work_usd"]) == (1.25, 2.0)
    _assert_fenced(ledger)


def test_cleanup_failure_during_paid_retry_preserves_previous_cost_history(tmp_path, owned):
    _provider, sandbox, _client, _remove = owned
    ledger = Ledger(tmp_path)
    ledger.write(CellResult(task="first", arm="A", trial=1, status="infra", config_hash="original",
                            billed_usd=0.75, work_usd=1.0))
    original = ledger.path_for(Cell("first", "A", 1)).read_bytes()

    def execute(cell, context):
        sandbox.close()
        return _run_result()

    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        run_sweep(tasks=["first"], arms=["A"], trials=1, retry_budget=1, ledger_dir=tmp_path,
                  prepare=lambda cell: SimpleNamespace(llm_stats=SimpleNamespace(
                      total_cost_usd=1.25, total_work_usd=1.25)), execute=execute)
    assert ledger.path_for(Cell("first", "A", 1)).read_bytes() == original
    _directory, receipt = _resources_receipt(tmp_path)
    assert receipt["attempts_made"] == 2 and receipt["billed_usd"] == 1.25
    _assert_fenced(tmp_path)


def test_fallible_cost_accessor_does_not_erase_cleanup_evidence(tmp_path, owned):
    _provider, sandbox, _client, _remove = owned

    class BrokenStats:
        @property
        def total_cost_usd(self):
            raise RuntimeError("cost accessor failed")

    def execute(cell, context):
        sandbox.close()
        return _run_result()

    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        run_sweep(tasks=["first"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: SimpleNamespace(llm_stats=BrokenStats()), execute=execute)
    _directory, receipt = _resources_receipt(tmp_path)
    assert receipt["billed_usd"] is None and receipt["work_usd"] is None
    assert receipt["resources"][0]["resource_id"] == sandbox.container.id
    _assert_fenced(tmp_path)


@pytest.mark.parametrize("mask", ["later_exception", "interrupt_group"])
def test_later_finally_or_group_cannot_hide_resource_failure(tmp_path, owned, mask):
    _provider, sandbox, _client, _remove = owned

    def execute(cell, context):
        try:
            sandbox.close()
        except Exception as exc:
            if mask == "interrupt_group":
                raise BaseExceptionGroup("shutdown", [KeyboardInterrupt(), exc]) from None
            raise ValueError("later metadata failed") from None
        return _run_result()

    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        run_sweep(tasks=["first"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: None, execute=execute)
    _directory, receipt = _resources_receipt(tmp_path)
    assert receipt["resources"][0]["resource_id"] == sandbox.container.id
    _assert_fenced(tmp_path)


def test_grading_recovery_records_new_cleanup_failure_before_any_further_retry(tmp_path, owned):
    _provider, sandbox, _client, _remove = owned

    def fail_grade(*args):
        raise RuntimeError("original grading failed")

    with pytest.raises(UnsettledSweepAttempt, match="grading is incomplete"):
        run_sweep(tasks=["first"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: None, execute=lambda cell, ctx: _run_result(), score=fail_grade)

    def recover(record):
        sandbox.close()
        return GradingResult(1.0, "report.json")

    with pytest.raises(UnsettledSweepAttempt, match="resource cleanup"):
        resume_grading(ledger_dir=tmp_path, score=recover)
    directory, receipt = _resources_receipt(tmp_path)
    assert receipt["phase"] == "score" and (directory / "execution.json").exists()
    _assert_fenced(tmp_path)
