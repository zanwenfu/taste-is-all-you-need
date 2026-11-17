"""Tests for the three memory designs (studies.claude_code.git_memory), the git memory's hooks and its MCP tool server: the same
items in every design, the store answering from its own commits, delivery once per touch until a compaction, and the service
reached through a real socket."""

import io
import json
import threading
import time
from pathlib import Path

import pytest

from studies.claude_code import git_memory as gm
from studies.claude_code.hooks import git_memory_hook as hook
from studies.claude_code.hooks import memory_mcp as mcp

MARC = "openlibrary/catalog/marc"
ITEMS = [
    {"kind": "folder", "scope": MARC, "tests": f"{MARC}/tests",
     "text": f"When you change code under {MARC}/, run its tests with: python -B -m pytest {MARC}/tests -q -p no:cacheprovider"},
    {"kind": "folder", "scope": "openlibrary/solr", "tests": "openlibrary/tests/solr",
     "text": "When you change code under openlibrary/solr/, run its tests with: python -B -m pytest openlibrary/tests/solr -q -p no:cacheprovider"},
    {"kind": "note", "scope": MARC, "file": f"{MARC}/parse.py", "text": f"{MARC}/parse.py: function read_edition(rec): Parse a MARC record."},
    {"kind": "note", "scope": "openlibrary/solr", "file": "openlibrary/solr/update.py",
     "text": "openlibrary/solr/update.py: function update_keys(keys): Send changed keys to the search index."},
]


def test_the_flat_and_folder_designs_hold_exactly_the_same_items():
    items = gm.with_rules(ITEMS)
    flat = gm.flat_markdown(items)
    assert all(f"- {i['text']}\n" in flat for i in items) and len(items) == len(ITEMS) + 8
    claude_md, files = gm.folder_rules(items)
    assert all(rule in claude_md for rule in gm.HOUSE_RULES) and "parse.py" not in claude_md
    assert sorted(i["text"] for i in items if i["kind"] != "rule") == sorted(
        line[2:] for text in files.values() for line in text.splitlines() if line.startswith("- "))
    assert files[".claude/rules/openlibrary__catalog__marc--tests.md"].startswith(f'---\npaths:\n  - "{MARC}/**"\n---\n')
    assert files[".claude/rules/openlibrary__catalog__marc--notes.md"].startswith(f'---\npaths:\n  - "{MARC}/*"\n---\n')


@pytest.fixture
def built(tmp_path):
    items = tmp_path / "items.json"
    items.write_text(json.dumps(ITEMS))
    summary = gm.build(items, tmp_path / "mem")
    return tmp_path / "mem", summary


def test_the_git_store_has_one_commit_per_item_and_answers_from_it(built):
    folder, summary = built
    assert summary["commits"] == len(ITEMS) + 8 and (folder / "flat" / "CLAUDE.md").exists()
    assert (folder / "folders" / ".claude" / "rules" / "openlibrary__solr--tests.md").exists()
    memory = gm.Memory(folder / "git")
    try:
        assert memory.rules() == list(gm.HOUSE_RULES)
        got = memory.items_for([f"/app/{MARC}/parse.py"])
        assert [i["kind"] for i in got] == ["folder", "note"] and "read_edition" in got[1]["text"]
        assert memory.items_for([f"{MARC}/tests"]) == [got[0]]  # a folder path brings its instruction, not a file's notes
        assert memory.items_for(["openlibrary/core/models.py"]) == []
        text, failed = memory.tool("memory_search", {"description": "send changed keys to the search index"})
        results = json.loads(text)["results"]
        assert not failed and "update_keys" in results[0]["snippet"] + json.dumps(results[0].get("header", ""))
        fetched, failed = memory.tool("memory_fetch", {"pointer": results[0]["pointer"]})
        assert not failed and "update_keys" in json.loads(fetched)["text"]
        assert {t["name"] for t in memory.definitions()} == {"memory_search", "memory_fetch"}
        assert memory.tool("memory_search", {"description": ""})[1] is True  # a bad call is an error the agent reads
    finally:
        memory.store.close()


def test_the_service_refuses_an_index_that_disagrees_with_the_store(built):
    folder, _ = built
    index = json.loads((folder / "git" / "index.json").read_text())
    for row in index:
        row["text"] += " (edited)"
    (folder / "git" / "index.json").write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="disagree"):
        gm.Memory(folder / "git")


@pytest.mark.parametrize("event, paths", [
    ({"tool_name": "Read", "tool_input": {"file_path": f"/app/{MARC}/parse.py"}}, [f"{MARC}/parse.py"]),
    ({"tool_name": "Bash", "tool_input": {"command": f"cd /app && python -B -m pytest {MARC}/tests -q"}}, [f"{MARC}/tests"]),
    ({"tool_name": "Grep", "tool_input": {"pattern": "def read", "path": "/app/openlibrary/solr"}}, ["openlibrary/solr"]),
    ({"tool_name": "Bash", "tool_input": {"command": "ls -la /tmp && echo done"}}, []),
])
def test_the_paths_a_tool_call_touches_are_read_from_its_arguments(event, paths):
    assert hook.touched_paths(event) == paths


class FakeService:
    def __init__(self, down=False):
        self.down, self.calls = down, []

    def __call__(self, memory, request, timeout=30):
        self.calls.append(request)
        if self.down:
            raise ConnectionRefusedError("no service")
        if request["op"] == "rules":
            return {"rules": ["Rule A.", "Rule B."]}
        paths = request.get("paths", [])
        return {"items": [{"id": 1, "kind": "folder", "text": "run the marc tests"}, {"id": 2, "kind": "note", "text": "parse.py note"}]
                if any("marc" in p for p in paths) else []}


def _hook(mode, event, memory, monkeypatch, service):
    monkeypatch.setattr(hook, "ask", service)
    out = io.StringIO()
    hook.main([mode, "--memory", str(memory)], stdin=io.StringIO(json.dumps(event)), stdout=out)
    return json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"] if out.getvalue() else None


def test_rules_go_with_every_message_and_a_folder_comes_once_until_a_compaction(tmp_path, monkeypatch):
    service = FakeService()
    rules = _hook("prompt", {"prompt": "fix it", "session_id": "s"}, tmp_path, monkeypatch, service)
    assert rules.startswith(hook.RULES_HEADER) and "- Rule A.\n- Rule B." in rules and "memory_search" in rules
    touch = {"tool_name": "Read", "tool_input": {"file_path": f"/app/{MARC}/parse.py"}, "session_id": "s"}
    first = _hook("post", touch, tmp_path, monkeypatch, service)
    assert first.startswith(hook.TOUCH_HEADER) and "run the marc tests" in first and "parse.py note" in first
    assert _hook("post", touch, tmp_path, monkeypatch, service) is None  # already delivered in this context
    assert _hook("post", {"tool_name": "Read", "tool_input": {"file_path": "/app/openlibrary/core/models.py"}, "session_id": "s"},
                 tmp_path, monkeypatch, service) is None
    again = _hook("compact", {"source": "compact", "session_id": "s"}, tmp_path, monkeypatch, service)
    assert again.startswith(hook.RULES_HEADER)
    assert "run the marc tests" in _hook("post", touch, tmp_path, monkeypatch, service)  # after a compaction it comes again
    assert _hook("compact", {"source": "startup", "session_id": "s"}, tmp_path, monkeypatch, service) is None
    log = [json.loads(line) for line in (tmp_path / "deliveries.jsonl").read_text().splitlines()]
    assert [r["mode"] for r in log] == ["prompt", "post", "compact", "post"] and log[1]["folder_items"] == 1


def test_a_memory_service_that_is_down_never_blocks_claude(tmp_path, monkeypatch):
    assert _hook("prompt", {"prompt": "x"}, tmp_path, monkeypatch, FakeService(down=True)) is None
    assert "ConnectionRefusedError" in (tmp_path / "deliveries.jsonl").read_text()


def test_the_tool_server_speaks_mcp_and_passes_calls_to_the_service(tmp_path, monkeypatch):
    def fake(memory, request, timeout=30):
        if request["op"] == "definitions":
            return {"tools": [{"name": "memory_search", "description": "d", "inputSchema": {"type": "object"}}]}
        return {"text": '{"results": []}', "failed": request["arguments"].get("description") == ""}
    monkeypatch.setattr(mcp, "ask", fake)
    init = mcp.respond({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}, tmp_path)
    assert init["result"]["protocolVersion"] == "2025-06-18" and "tools" in init["result"]["capabilities"]
    assert mcp.respond({"jsonrpc": "2.0", "method": "notifications/initialized"}, tmp_path) is None
    assert mcp.respond({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, tmp_path)["result"]["tools"][0]["name"] == "memory_search"
    call = mcp.respond({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "memory_search", "arguments": {"description": "marc"}}}, tmp_path)
    assert call["result"] == {"content": [{"type": "text", "text": '{"results": []}'}], "isError": False}
    bad = mcp.respond({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "memory_search", "arguments": {"description": ""}}},
                      tmp_path)
    assert bad["result"]["isError"] is True
    assert mcp.respond({"jsonrpc": "2.0", "id": 5, "method": "nope"}, tmp_path)["error"]["code"] == -32601


def test_the_hook_and_the_tool_server_reach_the_real_store_through_a_socket(built, tmp_path):
    folder, _ = built
    memory = gm.Memory(folder / "git")
    sock = folder / "git" / "memory.sock"
    thread = threading.Thread(target=gm.serve, args=(memory, sock), daemon=True)
    thread.start()
    for _ in range(100):
        if sock.exists():
            break
        time.sleep(0.05)
    try:
        assert hook.ask(folder / "git", {"op": "rules"})["rules"] == list(gm.HOUSE_RULES)
        out = io.StringIO()
        hook.main(["post", "--memory", str(folder / "git")], stdout=out, stdin=io.StringIO(json.dumps(
            {"tool_name": "Edit", "tool_input": {"file_path": f"/app/{MARC}/parse.py"}, "session_id": "s1"})))
        assert "pytest openlibrary/catalog/marc/tests" in out.getvalue() and "read_edition" in out.getvalue()
        lines = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                        "params": {"name": "memory_search", "arguments": {"description": "MARC record"}}}) + "\n")
        out = io.StringIO()
        mcp.main(["--memory", str(folder / "git")], stdin=lines, stdout=out)
        reply = json.loads(out.getvalue())
        assert reply["result"]["isError"] is False and "read_edition" in reply["result"]["content"][0]["text"]
        log = [json.loads(line) for line in (folder / "git" / "service.jsonl").read_text().splitlines()]
        assert [r["op"] for r in log] == ["rules", "items_for", "tool"]
    finally:
        memory.store.close()


FACT = {"kind": "fact", "scope": "scripts/sitemaps", "text": "Project note (sitemaps): the sitemaps script must run with --chunk 45000."}


def test_project_facts_are_in_every_design_and_come_with_their_folder(tmp_path):
    items = gm.with_rules(ITEMS + [FACT])
    assert "## Project notes\n\n- " + FACT["text"] in gm.flat_markdown(items)
    _, files = gm.folder_rules(items)
    assert files[".claude/rules/scripts__sitemaps--facts.md"].startswith('---\npaths:\n  - "scripts/sitemaps/**"\n---\n')
    (tmp_path / "items.json").write_text(json.dumps(ITEMS + [FACT]))
    gm.build(tmp_path / "items.json", tmp_path / "mem")
    memory = gm.Memory(tmp_path / "mem" / "git")
    try:
        assert [i["text"] for i in memory.items_for(["/app/scripts/sitemaps/sitemap.py"])] == [FACT["text"]]
        text, _ = memory.tool("memory_search", {"description": "what chunk size for the sitemaps script"})
        assert "45000" in text
    finally:
        memory.store.close()


def test_with_refresh_a_folder_instruction_comes_again_with_every_task_and_notes_do_not(tmp_path, monkeypatch):
    service = FakeService()
    touch = {"tool_name": "Read", "tool_input": {"file_path": f"/app/{MARC}/parse.py"}, "session_id": "s"}

    def hook_run(mode, event, refresh):
        monkeypatch.setattr(hook, "ask", service)
        out = io.StringIO()
        hook.main([mode, "--memory", str(tmp_path)] + (["--refresh-folders"] if refresh else []), stdin=io.StringIO(json.dumps(event)),
                  stdout=out)
        return json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"] if out.getvalue() else ""

    hook_run("prompt", {"prompt": "task 1", "session_id": "s"}, True)
    first = hook_run("post", touch, True)
    assert "run the marc tests" in first and "parse.py note" in first
    assert hook_run("post", touch, False) == ""  # within the same task, nothing again
    hook_run("prompt", {"prompt": "task 2", "session_id": "s"}, True)
    second = hook_run("post", touch, True)
    assert "run the marc tests" in second and "parse.py note" not in second  # the instruction again, the notes not
    hook_run("prompt", {"prompt": "task 3", "session_id": "s"}, False)  # without the option nothing is forgotten
    assert hook_run("post", touch, False) == ""


def test_claude_saves_each_entry_as_its_own_commit_and_the_reader_does_not_repeat_it(tmp_path):
    (tmp_path / "items.json").write_text(json.dumps([i for i in ITEMS if i["kind"] == "note"]))
    gm.build(tmp_path / "items.json", tmp_path / "mem", house_rules=False)
    memory = gm.Memory(tmp_path / "mem" / "git")
    try:
        assert memory.rules() == []
        text, failed = memory.tool("memory_save", {"kind": "rule", "text": "Keep hand-backs under 150 words."})
        assert not failed and json.loads(text)["saved"] == 2
        memory.tool("memory_save", {"kind": "decision", "text": "Use the ISBNdb name everywhere.", "scope": "scripts/providers"})
        memory.tool("memory_save", {"kind": "fact", "text": "The sitemaps script runs with --chunk 45000."})
        assert memory.rules() == ["Keep hand-backs under 150 words.", "Use the ISBNdb name everywhere."]  # facts are searched, not sent
        assert memory.handle({"op": "save", "kind": "rule", "source": "reader", "text": "keep your hand-backs under 150 words please"}
                             )["skipped"] == "already kept"
        assert "saved" in memory.handle({"op": "save", "kind": "rule", "source": "reader", "text": "Never use git stash."})
        assert "saved" in memory.handle({"op": "save", "kind": "turn", "source": "hook", "text": "Turn record. Task: fix the parser"})
        assert memory.tool("memory_save", {"kind": "turn", "text": "x"})[1] is True  # Claude saves rules, decisions and facts only
        assert memory.tool("memory_save", {"kind": "fact", "text": ""})[1] is True
        assert "45000" in memory.tool("memory_search", {"description": "sitemaps chunk"})[0]
        assert {t["name"] for t in memory.definitions(saving=True)} == {"memory_search", "memory_fetch", "memory_save"}
        assert "this session" not in json.dumps(memory.definitions())
    finally:
        memory.store.close()
    again = gm.Memory(tmp_path / "mem" / "git")  # the saves are on disk and agree with the store
    try:
        assert [r["kind"] for r in again.index][-5:] == ["rule", "decision", "fact", "rule", "turn"]
        assert again.rules()[-1] == "Never use git stash."
    finally:
        again.store.close()


def _transcript(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_the_last_turn_is_read_from_claude_codes_own_transcript(tmp_path):
    path = _transcript(tmp_path / "t.jsonl", [
        {"type": "user", "message": {"role": "user", "content": "first task"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done one"}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "From now on use python -B.\n\nTitle: Fix the parser"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": "python -B -m pytest x"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Files changed: a.py\n\nFixed it."}]}}])
    turn = hook.last_turn(path)
    assert turn["user"].startswith("From now on use python -B.") and turn["reply"].startswith("Files changed: a.py")
    assert turn["commands"] == ["python -B -m pytest x"] and hook.task_title(turn["user"]) == "Fix the parser"


def test_the_stop_hook_records_the_turn_and_the_reader_backs_up_claudes_saves(tmp_path, monkeypatch):
    import subprocess as sp
    project = tmp_path / "app"
    project.mkdir()
    sp.run(["git", "init", "-q", str(project)], check=True)
    (project / "a.py").write_text("x = 1\n")
    (project / "CLAUDE.md").write_text("# notes\n")
    path = _transcript(tmp_path / "t.jsonl", [
        {"type": "user", "message": {"content": "From now on use python -B.\n\nTitle: Fix the parser"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Fixed."}]}}])
    calls = []
    monkeypatch.setattr(hook, "ask", lambda memory, request, timeout=30: calls.append(request) or {"saved": len(calls)})
    import types
    fake = types.ModuleType("studies.claude_code.readers")
    fake.claude_reader = lambda message: ["From now on use python -B."]
    monkeypatch.setitem(__import__("sys").modules, "studies.claude_code.readers", fake)
    out = hook.run("stop", {"transcript_path": str(path), "session_id": "s", "last_assistant_message": "Files changed: a.py. Done."},
                   tmp_path, live=True, reader=True, project=project)
    assert out == ""  # a Stop hook that prints nothing never blocks Claude
    assert calls[0]["kind"] == "turn" and "Task: Fix the parser" in calls[0]["text"] and "a.py" in calls[0]["text"]
    assert "Claude's reply: Files changed: a.py. Done." in calls[0]["text"]  # the reply Claude Code hands the hook, not the transcript's
    assert "CLAUDE.md" not in calls[0]["text"]  # the other designs' memory files are not the task's changes
    assert calls[1] == {"op": "save", "kind": "rule", "source": "reader", "text": "From now on use python -B."}
    assert hook.run("stop", {"transcript_path": str(path), "stop_hook_active": True}, tmp_path, reader=True, project=project) == ""
    assert len(calls) == 2


def test_the_live_message_says_how_to_save_and_brings_what_was_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "ask", lambda memory, request, timeout=30: {"rules": []})
    empty = hook.rules_text(tmp_path, live=True)
    assert empty == hook.LIVE_TOOLS and "memory_save" in empty
    monkeypatch.setattr(hook, "ask", lambda memory, request, timeout=30: {"rules": ["Use python -B."]})
    assert hook.rules_text(tmp_path, live=True).startswith(hook.LIVE_HEADER + "\n- Use python -B.\n")
    seen = []
    monkeypatch.setattr(mcp, "ask", lambda memory, request, timeout=30: seen.append(request) or {"tools": []})
    mcp.respond({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, tmp_path, saving=True)
    assert seen[-1] == {"op": "definitions", "saving": True}


def test_a_runs_copy_of_the_memory_can_save_and_the_source_is_untouched(tmp_path):
    (tmp_path / "items.json").write_text(json.dumps([i for i in ITEMS if i["kind"] == "note"]))
    gm.build(tmp_path / "items.json", tmp_path / "built", house_rules=False)
    source = gm.Memory(tmp_path / "built" / "git")  # the source's own checkout exists and stays registered, as on the machine
    source.save("rule", "Seed rule.")
    before = (tmp_path / "built" / "git" / "index.json").read_text()
    source.store.close()
    gm.copy_memory(tmp_path / "built" / "git", tmp_path / "run" / "memory")
    copy = gm.Memory(tmp_path / "run" / "memory")
    try:
        assert "saved" in copy.handle({"op": "save", "kind": "turn", "source": "hook", "text": "Turn record. Task: x"})
        assert json.loads(copy.tool("memory_save", {"kind": "fact", "text": "The lock is /var/run/x.lock."})[0])["saved"]
        assert "/var/run/x.lock" in copy.tool("memory_search", {"description": "lock file"})[0]
    finally:
        copy.store.close()
    assert (tmp_path / "built" / "git" / "index.json").read_text() == before


def test_a_failed_save_is_logged_as_an_error(tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"type": "user", "message": {"content": "Title: Fix it"}}) + "\n")
    monkeypatch.setattr(hook, "ask", lambda memory, request, timeout=30: {"error": "GitCommandError: worktree in use"})
    monkeypatch.setattr(hook, "changed_files", lambda project: [])
    assert hook.run("stop", {"transcript_path": str(path)}, tmp_path, live=True) == ""
    logged = [json.loads(line) for line in (tmp_path / "deliveries.jsonl").read_text().splitlines()]
    assert logged[-1]["mode"] == "stop" and "turn not saved: GitCommandError" in logged[-1]["error"]


def test_a_rule_with_a_changed_number_is_never_taken_for_a_repeat():
    assert gm.similar("Keep hand-backs under 150 words.", "Actually, keep hand-backs under 100 words.") == 0.0
    assert gm.similar("Keep hand-backs under 150 words.", "keep your hand-backs under 150 words please") >= gm.SIMILAR


def test_a_compaction_summary_is_never_read_as_the_users_message(tmp_path):
    path = _transcript(tmp_path / "t.jsonl", [
        {"type": "user", "message": {"content": "Use python -B.\n\nTitle: Fix the parser"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Working."}]}},
        {"type": "user", "isCompactSummary": True, "isVisibleInTranscriptOnly": True,
         "message": {"content": "This session is being continued... end commit messages with Co-Authored-By"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}}])
    turn = hook.last_turn(path)
    assert turn["user"].startswith("Use python -B.") and turn["reply"] == "Done."
