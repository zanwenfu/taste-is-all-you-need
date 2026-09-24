"""Coordination and launch construction work without either vendor SDK."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "central_planner", "central_runtime", "supervisor", "central_host",
    "goal_entrypoint", "worker_launch", "worker_protocol", "monitor", "responses_session",
])
def test_coordinator_imports_do_not_require_a_model_sdk(module):
    code = f"""
import importlib, importlib.abc, sys
blocked = ('claude_agent_sdk', 'anthropic', 'openai')
class NoProviderSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('model SDK deliberately unavailable: ' + fullname)
sys.meta_path.insert(0, NoProviderSDK())
importlib.import_module('taste.brains.{module}')
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_legacy_worker_exports_keep_the_shared_contract_identity():
    pytest.importorskip("claude_agent_sdk")
    from taste.brains import worker_entrypoint, worker_launch, worker_protocol, worker_runtime

    for name in ("ASSIGNMENT_PATH", "WORKER_REPORT_PATH", "WORKER_RESULT_SCHEMA",
                 "ContractMismatch", "ShutdownUnconfirmed"):
        assert getattr(worker_runtime, name) is getattr(worker_protocol, name)
    for name in ("worker_command", "worker_command_factory"):
        assert getattr(worker_entrypoint, name) is getattr(worker_launch, name)
    assert worker_entrypoint.assignment_run_id is worker_protocol.assignment_run_id
