"""The same exact launch boundary is usable without the Claude SDK."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.records import ArtifactRef, ArtifactSpec, Assignment, contract_digest
from taste.brains.worker_admission import (
    EntrypointConfig,
    EntrypointInputError,
    _load_durable_input,
    validate_acquired_branch,
)
from taste.brains.worker_protocol import ASSIGNMENT_PATH, assignment_run_id
from taste.memstore import Store
from taste.providers.azure_openai import AZURE_MONITOR_MODEL, AZURE_WORKER_MODEL


@pytest.fixture
def launch(tmp_path):
    store = Store.open(tmp_path / "repo", "admission")
    producer = store.branch("producer")
    producer.write("input.txt", "exact input")
    source = producer.checkpoint("input source")
    producer.close()
    branch = store.branch("worker")
    brief = Contract("worker", "produce output", inputs=("input.txt",), outputs=("output.txt",),
                     success_criteria=("the output is correct",), budget_usd=30)
    assignment = Assignment(
        "task", 1, 0, brief, contract_digest(brief), branch.head.id,
        inputs=(ArtifactRef("input", "producer", source.id, "input.txt", source.blob("input.txt")),),
        outputs=(ArtifactSpec("output", "output.txt"),), model=AZURE_WORKER_MODEL,
        resources={"monitor_budget_usd": 30},
    )
    branch.write("input.txt", source.read_bytes("input.txt"))
    branch.write(CONTRACT_PATH, brief.to_json())
    branch.write(ASSIGNMENT_PATH, assignment.to_json())
    prepared = branch.checkpoint("prepared assignment")
    branch.close()
    config = EntrypointConfig(store.root, store.session, "worker", AZURE_WORKER_MODEL,
                              prepared.id, monitor_model=AZURE_MONITOR_MODEL, monitor_budget_usd=30)
    run_id = assignment_run_id(assignment)
    key = hashlib.sha256(run_id.encode()).hexdigest()
    environ = {"TASTE_WORKER_RUN_ID": run_id, "TASTE_WORKER_LAUNCH_TOKEN": "a" * 32,
               "TASTE_WORKER_READY_PATH": str(store.backend.common_dir / f"taste-supervisor.admission.{key}.ready.json")}
    yield SimpleNamespace(store=store, assignment=assignment, config=config, environ=environ)
    store.close()


def admit(launch):
    return _load_durable_input(launch.store, launch.config, launch.environ)


def install_assignment(launch, assignment):
    branch = launch.store.branch("worker")
    branch.write(ASSIGNMENT_PATH, assignment.to_json())
    branch.write(CONTRACT_PATH, assignment.contract.to_json())
    state = branch.checkpoint("changed prepared assignment")
    branch.close()
    launch.config = replace(launch.config, prepared_state_id=state.id)
    run_id = assignment_run_id(assignment)
    launch.environ["TASTE_WORKER_RUN_ID"] = run_id
    key = hashlib.sha256(run_id.encode()).hexdigest()
    launch.environ["TASTE_WORKER_READY_PATH"] = str(
        launch.store.backend.common_dir / f"taste-supervisor.admission.{key}.ready.json")


def test_exact_azure_assignment_is_read_without_a_lease_then_revalidated_under_it(launch):
    durable = admit(launch)
    assert durable.assignment == launch.assignment
    branch = launch.store.branch("worker")
    validate_acquired_branch(branch, launch.store, launch.config, durable)
    assert branch.read("input.txt") == "exact input"


@pytest.mark.parametrize("budget", [0, 0.0, 30, 30.0, 0.25])
def test_budget_has_one_canonical_representation_across_contract_and_assignment(launch, budget):
    brief = replace(launch.assignment.contract, budget_usd=budget)
    assignment = replace(launch.assignment, contract=brief, contract_digest=contract_digest(brief))
    decoded = Assignment.from_json(assignment.to_json())
    assert decoded.contract.to_json() == brief.to_json()
    assert Contract.from_json(brief.to_json()).to_json() == brief.to_json()


@pytest.mark.parametrize("field,value", [
    ("TASTE_WORKER_RUN_ID", "worker-run." + "0" * 64),
    ("TASTE_WORKER_LAUNCH_TOKEN", "short"),
    ("TASTE_WORKER_READY_PATH", "/tmp/unbound-ready.json"),
])
def test_unbound_supervisor_environment_is_rejected(launch, field, value):
    launch.environ[field] = value
    with pytest.raises(EntrypointInputError):
        admit(launch)


@pytest.mark.parametrize("damage", ["wrong_model", "wrong_branch", "missing_source", "unrelated_base"])
def test_provider_independent_admission_preserves_runtime_input_provenance_checks(launch, damage):
    assignment = launch.assignment
    if damage == "wrong_model":
        assignment = replace(assignment, model="claude-sonnet-5")
    elif damage == "wrong_branch":
        assignment = replace(assignment, inputs=(replace(assignment.inputs[0], branch="someone-else"),))
    elif damage == "missing_source":
        assignment = replace(assignment, inputs=(replace(assignment.inputs[0], state_id="f" * 40),))
    else:
        other = launch.store.branch("unrelated")
        other.write("separate.txt", "other work")
        state = other.checkpoint("not an ancestor of worker")
        other.close()
        assignment = replace(assignment, base_state_id=state.id)
    install_assignment(launch, assignment)
    with pytest.raises(EntrypointInputError):
        admit(launch)


def test_source_from_another_session_is_rejected_even_when_bytes_match(launch):
    other = Store.open(launch.store.root, "other-session")
    try:
        producer = other.branch("producer")
        producer.write("input.txt", "exact input")
        source = producer.checkpoint("same bytes, different session")
        ref = replace(launch.assignment.inputs[0], state_id=source.id)
        install_assignment(launch, replace(launch.assignment, inputs=(ref,)))
        with pytest.raises(EntrypointInputError, match="another session"):
            admit(launch)
    finally:
        other.close()


@pytest.mark.parametrize("damage", ["dirty", "head_changed"])
def test_change_between_read_and_lease_acquisition_is_rejected(launch, damage):
    durable = admit(launch)
    branch = launch.store.branch("worker")
    branch.write("input.txt", "changed after exact read")
    if damage == "head_changed":
        branch.checkpoint("another controller wrote before acquisition")
    with pytest.raises(EntrypointInputError):
        validate_acquired_branch(branch, launch.store, launch.config, durable)


@pytest.mark.parametrize("path", [CONTRACT_PATH, ASSIGNMENT_PATH, "input.txt"])
def test_mode_changes_are_rejected_even_when_control_and_input_bytes_match(launch, path):
    branch = launch.store.branch("worker")
    branch.path(path).chmod(0o755)
    state = branch.checkpoint("same bytes, altered mode")
    branch.close()
    launch.config = replace(launch.config, prepared_state_id=state.id)
    with pytest.raises(EntrypointInputError):
        admit(launch)


def test_shared_admission_requires_no_claude_sdk_in_a_fresh_process():
    script = '''
import importlib.abc, sys
class DenyClaude(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] == 'claude_agent_sdk':
            raise AssertionError('shared worker admission imported Claude')
sys.meta_path.insert(0, DenyClaude())
from taste.brains.worker_admission import EntrypointConfig, _load_durable_input, validate_acquired_branch
assert not any(name.startswith('claude_agent_sdk') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
