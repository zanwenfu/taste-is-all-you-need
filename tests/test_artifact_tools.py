"""Confined file effects through the real Responses turn and memory boundary."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from types import SimpleNamespace

import pytest

from taste.brains.artifact_tools import SAVED, ArtifactTools
from taste.brains.contract import Contract
from taste.brains.records import ArtifactRef, ArtifactSpec, Assignment, contract_digest
from taste.brains.responses_conversation import ResponsesConversation
from taste.brains.responses_session import ResponsesFenced, ResponsesSession
from taste.memstore import Store
from taste.providers.base import ToolCall
from tests.test_azure_openai import config
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_openai_responses import function_call, message
from tests.test_responses_conversation import install
from tests.test_responses_session import binding

sdk_transport = _sdk_transport


@pytest.fixture
def worker(tmp_path):
    store = Store.open(tmp_path / "repo", "artifacts")
    branch = store.branch("worker")
    branch.write("input.txt", "immutable input")
    source = branch.checkpoint("input")
    contract = Contract("worker", "produce outputs", inputs=("input.txt",),
                        outputs=("output.txt", "nested/output.txt"),
                        success_criteria=("outputs are correct",))
    assignment = Assignment(
        "task", 1, 0, contract, contract_digest(contract), source.id,
        inputs=(ArtifactRef("input", "worker", source.id, "input.txt", source.blob("input.txt")),),
        outputs=(ArtifactSpec("output", "output.txt"), ArtifactSpec("nested", "nested/output.txt")),
        model=binding().model,
    )
    yield SimpleNamespace(store=store, branch=branch, assignment=assignment,
                          files=ArtifactTools(branch, assignment))
    store.close()


def write_args(path="output.txt", content="result", executable=False):
    return {"artifact": path, "body": content, "executable": executable}


def read_args(path="output.txt", offset=0, limit=8192):
    return {"artifact": path, "offset": offset, "limit": limit}


def invoke(worker, name, arguments):
    call = ToolCall("call", name, arguments)
    return asyncio.run(worker.files.tools()[name].execute("effect", call))


def test_nested_atomic_write_read_mode_and_removal(worker):
    assert not invoke(worker, "write_artifact", write_args("nested/output.txt", "π result", True)).is_error
    path = worker.branch.path("nested/output.txt")
    assert path.stat().st_mode & 0o777 == 0o755
    result = invoke(worker, "read_artifact", read_args("nested/output.txt"))
    assert json.loads(result.content) == {"content": "π result", "encoding": "utf-8", "size": 9,
                                        "next_offset": 9, "eof": True, "executable": True}
    assert not invoke(worker, "write_artifact", write_args("nested/output.txt", "new")).is_error
    assert path.stat().st_mode & 0o777 == 0o644
    assert not invoke(worker, "remove_artifact", {"artifact": "nested/output.txt"}).is_error
    assert not path.exists()
    assert not invoke(worker, "remove_artifact", {"artifact": "nested/output.txt"}).is_error
    assert not list(path.parent.glob(".taste-artifact-*"))


@pytest.mark.parametrize("arguments", [
    write_args("input.txt"), write_args("../outside"), write_args(".git/config"),
    write_args("/tmp/outside"), write_args("nested/../output.txt"), write_args("output.txt/child"),
    write_args(content="x" * 65_537), write_args(executable=1),
    {**write_args(), "extra": True}, {"artifact": "output.txt"}, write_args(content="\ud800"),
])
def test_invalid_write_is_rejected_before_any_effect(worker, arguments):
    with pytest.raises(ValueError):
        invoke(worker, "write_artifact", arguments)
    assert worker.branch.read("input.txt") == "immutable input"
    assert not worker.branch.path("output.txt").exists()


@pytest.mark.parametrize("arguments", [read_args(offset=-1), read_args(offset=True),
                                        read_args(limit=0), read_args(limit=8193), read_args(limit=True),
                                        read_args(path="unassigned.txt")])
def test_invalid_read_is_rejected(worker, arguments):
    with pytest.raises(ValueError):
        invoke(worker, "read_artifact", arguments)


@pytest.mark.parametrize("damage", ["parent_symlink", "leaf_symlink", "hardlink", "fifo", "directory"])
def test_special_files_and_links_cannot_escape_or_block(worker, tmp_path, damage):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "output.txt"
    target.write_text("protected")
    relative = "nested/output.txt" if damage == "parent_symlink" else "output.txt"
    leaf = worker.branch.path(relative)
    if damage == "parent_symlink":
        worker.branch.path("nested").symlink_to(outside, target_is_directory=True)
    elif damage == "leaf_symlink":
        leaf.symlink_to(target)
    elif damage == "hardlink":
        os.link(target, leaf)
    elif damage == "fifo":
        os.mkfifo(leaf)
    else:
        leaf.mkdir()
    assert invoke(worker, "read_artifact", read_args(relative)).is_error
    assert invoke(worker, "write_artifact", write_args(relative)).is_error
    assert target.read_text() == "protected"
    result = invoke(worker, "remove_artifact", {"artifact": relative})
    assert result.is_error == (damage in {"parent_symlink", "fifo", "directory"})
    assert target.read_text() == "protected"


def test_binary_and_partial_utf8_paging_remain_bounded(worker):
    payload = "π".encode() + b"\x00" * 9000 + b"\xff"
    worker.branch.write("output.txt", payload)
    first = json.loads(invoke(worker, "read_artifact", read_args(limit=1)).content)
    assert first["encoding"] == "base64" and base64.b64decode(first["content"]) == payload[:1]
    assert first["next_offset"] == 1 and not first["eof"]
    second = invoke(worker, "read_artifact", read_args(offset=2))
    assert len(second.content.encode()) < 65_536
    assert len(json.loads(second.content)["content"]) == 8192
    end = json.loads(invoke(worker, "read_artifact", read_args(offset=len(payload))).content)
    assert end["content"] == "" and end["eof"]
    assert not invoke(worker, "write_artifact", write_args(content="x" * 65_536)).is_error


@pytest.mark.parametrize("damage", ["released", "reacquired", "different_process", "replaced_root"])
def test_tools_keep_the_original_lease_and_worktree_identity(worker, monkeypatch, damage):
    if damage in {"released", "reacquired"}:
        worker.branch.release()
        if damage == "reacquired":
            worker.store.branch("worker")
    elif damage == "different_process":
        monkeypatch.setattr(worker.files, "_pid", os.getpid() + 1)
    else:
        original = worker.branch.worktree
        original.rename(original.with_name(original.name + "-old"))
        original.mkdir()
    if damage == "replaced_root":
        assert invoke(worker, "write_artifact", write_args()).is_error
    else:
        with pytest.raises(ValueError, match="active branch lease"):
            invoke(worker, "write_artifact", write_args())
    assert not worker.branch.path("output.txt").exists()


def test_pending_cancellation_cannot_start_a_file_effect(worker):
    async def scenario():
        task = asyncio.current_task()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker.files.write("effect", ToolCall("call", "write_artifact", write_args()))
        # Consume the event loop's scheduled cancellation as well.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
    asyncio.run(scenario())
    assert not worker.branch.path("output.txt").exists()


def test_error_after_atomic_replace_reports_uncertainty_and_leaves_no_temporary(worker, monkeypatch):
    original = os.replace

    def replace_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("directory sync acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace_then_fail)
        result = invoke(worker, "write_artifact", write_args())
    assert result.is_error and "not confirmed" in result.content
    assert worker.branch.path("output.txt").read_text() == "result"
    assert worker.branch.head.read("output.txt") is None
    assert not list(worker.branch.worktree.glob(".taste-artifact-*"))


def test_real_tool_roundtrip_checkpoint_and_rollback_preserve_paid_receipts(worker, sdk_transport):
    sent = install(sdk_transport, [function_call(json.dumps(write_args()), name="write_artifact")],
                   [message("completed")])
    session = ResponsesSession.create(worker.store.backend.common_dir / "responses", binding(), config())
    before = worker.branch.head
    try:
        conversation = ResponsesConversation(worker.branch, session, system="Produce the artifact.", tools=worker.files.tools())
        conversation.observe("task", "Write output.txt")

        async def scenario():
            await conversation.step()
            saved = worker.branch.checkpoint("model reply and actual artifact effect")
            assert saved.read("output.txt") == "result"
            await conversation.step()
            worker.branch.checkpoint("completed")
            worker.branch.rollback(before, "restore work and reasoning")
            assert not worker.branch.path("output.txt").exists()
            assert session.known_cost_usd == pytest.approx(0.000732)

        asyncio.run(scenario())
        assert len(sent) == 2
        tool_result = [item for item in json.loads(sent[1].content)["input"]
                       if item.get("type") == "function_call_output"]
        # The result says which side the write landed on: a reader of the
        # record must not take it for a change to the task's repository.
        assert tool_result[0]["output"] == SAVED and "task environment" in SAVED
    finally:
        session.close()


def test_actual_write_with_lost_result_is_fenced_on_reopen(worker, sdk_transport, monkeypatch):
    sent = install(sdk_transport, [function_call(json.dumps(write_args()), name="write_artifact")])
    directory = worker.store.backend.common_dir / "responses"
    limits = binding()
    session = ResponsesSession.create(directory, limits, config())
    try:
        conversation = ResponsesConversation(worker.branch, session, system="Write output.", tools=worker.files.tools())
        conversation.observe("task", "Write output.txt")
        original = worker.branch.turn

        def lose_result(**event):
            if event.get("kind") == "responses_tool_result":
                raise OSError("receipt not persisted")
            original(**event)

        with monkeypatch.context() as patch:
            patch.setattr(worker.branch, "turn", lose_result)
            with pytest.raises(OSError):
                asyncio.run(conversation.step())
        assert worker.branch.path("output.txt").read_text() == "result"
        saved = worker.branch.checkpoint("unconfirmed applied effect")
        assert saved.read("output.txt") == "result"
    finally:
        session.close()
    session = ResponsesSession.open(directory, limits, config())
    try:
        reopened = ResponsesConversation(worker.branch, session, system="Write output.", tools=worker.files.tools())
        with pytest.raises(ResponsesFenced, match="unknown outcome"):
            asyncio.run(reopened.step())
        assert len(sent) == 1
    finally:
        session.close()
