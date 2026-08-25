"""The live console.

A finished report and a running one need different things. The property that
matters most here is that a run still in flight is never presented as a clean
one: probes are replayed *after* a run, so mid-run there is no regression
verdict, and an empty panel would read as "nothing broke".

The second is that a run is discovered by its event log rather than a
manifest. The log exists from the first moment of a run, so a run that died
during planning still appears — and that is precisely the run someone opens
the console to look at.

Hermetic: no server started except on an ephemeral port, no network.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from taste import server as server_mod
from taste.server import (
    Discovery,
    RunHandle,
    _status_from_events,
    read_events,
    run_payload,
)


def _events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _workspace(root: Path, instance: str, arm: str, trial: int, rows: list[dict]) -> Path:
    ws = root / "runs" / instance / arm / f"t{trial}"
    ws.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    (ws / "f.txt").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.co",
                    "commit", "-qm", "base"], cwd=ws, check=True)
    _events(ws / ".git" / "taste" / "events.jsonl", rows)
    return ws


RUNNING = [
    {"ts": 1.0, "kind": "run.start", "payload": {"session": "s1", "task": "do it", "branch": "b"}},
    {"ts": 1.1, "kind": "plan.ready", "payload": {"steps": 2}},
    {"ts": 1.2, "kind": "step.begin", "payload": {"id": "step-01", "attempt": 1}},
]
FINISHED = [
    *RUNNING,
    {"ts": 2.0, "kind": "run.done",
     "payload": {"status": "completed", "elapsed": 1.0, "cost_usd": 0.5}},
]


# ------------------------------------------------------------------ discovery


def test_a_run_is_found_by_its_event_log(tmp_path: Path) -> None:
    """The log exists from the first moment, so a run that died during
    planning still appears -- which is the run someone opens the console for."""
    _workspace(tmp_path, "inst-a", "A3", 1, RUNNING)
    found = Discovery(root=tmp_path).scan()

    assert len(found) == 1
    assert found[0].instance == "inst-a"
    assert found[0].arm == "A3"
    assert found[0].trial == 1


def test_a_finished_run_reports_its_own_last_word(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "inst-b", "A0", 2, FINISHED)
    assert _status_from_events(ws / ".git" / "taste" / "events.jsonl") == "completed"


def test_a_run_with_no_terminal_event_is_running(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "inst-c", "A2", 1, RUNNING)
    assert _status_from_events(ws / ".git" / "taste" / "events.jsonl") == "running"


def test_the_ledger_refines_the_status(tmp_path: Path) -> None:
    """The event log says "failed"; only the ledger knows it was the budget,
    and the distinction decides whether the cell is evidence."""
    _workspace(tmp_path, "inst-d", "A3", 1, FINISHED)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "inst-d__A3__t1.json").write_text(json.dumps({"status": "budget"}))

    assert Discovery(root=tmp_path).scan()[0].status == "budget"


# ------------------------------------------------------------------ tailing


def test_a_partially_written_line_is_left_for_next_time(tmp_path: Path) -> None:
    """A fast run is mid-write when the tail reads. Emitting the fragment
    sends truncated JSON to the browser."""
    log = tmp_path / "events.jsonl"
    log.write_text(json.dumps({"kind": "a"}) + "\n" + '{"kind": "par')

    events, offset = read_events(log, 0)

    assert [e["kind"] for e in events] == ["a"]
    log.write_text(json.dumps({"kind": "a"}) + "\n" + json.dumps({"kind": "b"}) + "\n")
    more, _ = read_events(log, offset)
    assert [e["kind"] for e in more] == ["b"], "the completed line must arrive next read"


def test_an_offset_resumes_rather_than_replaying(tmp_path: Path) -> None:
    """A reconnect must not replay a long run from the beginning."""
    log = tmp_path / "events.jsonl"
    _events(log, [{"kind": "a"}, {"kind": "b"}])
    _first, offset = read_events(log, 0)

    _events(log, [{"kind": "a"}, {"kind": "b"}, {"kind": "c"}])
    later, _ = read_events(log, offset)
    assert [e["kind"] for e in later] == ["c"]


def test_a_missing_log_is_empty_not_an_error(tmp_path: Path) -> None:
    assert read_events(tmp_path / "nope.jsonl", 0) == ([], 0)


def test_a_corrupt_line_does_not_stop_the_stream(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text('{"kind": "a"}\nnot json at all\n{"kind": "b"}\n')
    events, _ = read_events(log, 0)
    assert [e["kind"] for e in events] == ["a", "b"]


# ------------------------------------------------------------------ live


def test_a_running_run_says_regressions_are_not_yet_known(tmp_path: Path) -> None:
    """The property this file exists for. Probes replay after a run, so an
    empty regression panel mid-run would read as "nothing broke"."""
    ws = _workspace(tmp_path, "inst-e", "A3", 1, RUNNING)
    handle = RunHandle(run_id="x", workspace=ws, events=ws / ".git" / "taste" / "events.jsonl",
                       instance="inst-e", arm="A3", status="running")

    payload = run_payload(handle)

    assert payload["live"] is True
    joined = " ".join(payload["notes"])
    assert "not yet known" in joined
    assert "not known to be absent" in joined


def test_a_finished_run_is_not_marked_live(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "inst-f", "A3", 1, FINISHED)
    handle = RunHandle(run_id="x", workspace=ws, events=ws / ".git" / "taste" / "events.jsonl",
                       instance="inst-f", arm="A3", status="completed")
    assert run_payload(handle)["live"] is False


def test_the_payload_skips_diffs_because_the_console_repolls(tmp_path: Path) -> None:
    """A live run is re-read many times a minute; diffs belong in the report."""
    ws = _workspace(tmp_path, "inst-g", "A3", 1, FINISHED)
    handle = RunHandle(run_id="x", workspace=ws, events=ws / ".git" / "taste" / "events.jsonl",
                       instance="inst-g", arm="A3", status="completed")
    assert all(not o["diff"] for o in run_payload(handle)["observations"])


# ------------------------------------------------------------------ serving


def test_the_console_binds_to_localhost_by_default() -> None:
    """It serves file contents and command output from a machine holding API
    credentials, and it has no authentication."""
    import inspect

    from taste.server import serve

    assert inspect.signature(serve).parameters["host"].default == "127.0.0.1"


def test_the_server_answers_its_own_endpoints(tmp_path: Path) -> None:
    import threading
    import urllib.error
    import urllib.request

    from taste.server import serve

    _workspace(tmp_path, "inst-h", "A3", 1, FINISHED)
    httpd = serve(tmp_path, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
        assert b"taste console" in page

        runs = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=5).read()
        )
        assert len(runs) == 1 and runs[0]["instance"] == "inst-h"

        # An unknown run must 404 rather than 500 or hang.
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run/nope", timeout=5)
        assert caught.value.code == 404
    finally:
        httpd.shutdown()


# ============================================ the scan must not scale per run


def _sweep_root(tmp_path: Path, runs: int) -> Path:
    """A sweep root shaped like a real one: a ledger, and a checkout per cell.

    The decoy files matter. The defect this guards was invisible on a handful
    of empty directories and crippling on a real root, because the cost was
    (number of runs) x (number of files), and a real root carries a source
    tree and a git object store for every cell.
    """
    root = tmp_path / "sweep"
    ledger = root / "ledger" / "evidence"
    ledger.mkdir(parents=True)
    for i in range(runs):
        ws = root / "runs" / f"inst-{i}" / "A3" / "t1"
        taste_dir = ws / ".git" / "taste"
        taste_dir.mkdir(parents=True)
        (taste_dir / "events.jsonl").write_text(
            json.dumps({"kind": "run.done", "payload": {"status": "completed"}}) + "\n"
        )
        objects = ws / ".git" / "objects" / "ab"
        objects.mkdir(parents=True)
        for j in range(40):  # stand-in for a git object store
            (objects / f"{j:038x}").write_text("x")
        (root / "ledger" / f"inst-{i}__A3__t1.json").write_text(
            json.dumps({"status": "completed", "task": f"inst-{i}", "arm": "A3"})
        )
        (ledger / f"inst-{i}__A3__t1.json").write_text(json.dumps({"observations": 3}))
    return root


def test_scan_walks_the_tree_a_bounded_number_of_times(tmp_path: Path, monkeypatch) -> None:
    """One walk per scan, not one per run.

    Asserted by counting walks rather than by timing, so it states the
    invariant instead of measuring the machine. Before the fix this was one
    full-tree ``rglob`` per run: forty runs over a real sweep root took 36
    seconds to answer a single request, and the console -- which re-scans on
    every request -- exceeded every client timeout after its first response.
    """
    root = _sweep_root(tmp_path, runs=8)
    traversals = {"n": 0}
    real_walk, real_rglob = os.walk, Path.rglob

    # Both primitives are counted. An earlier version of this test counted
    # only `os.walk` and passed with the defect restored, because `rglob` does
    # not route through it -- a test that measured nothing and said so
    # confidently, which is the failure this whole project is about.
    def counting_walk(*args, **kwargs):
        traversals["n"] += 1
        return real_walk(*args, **kwargs)

    def counting_rglob(self, *args, **kwargs):
        traversals["n"] += 1
        return real_rglob(self, *args, **kwargs)

    monkeypatch.setattr(server_mod.os, "walk", counting_walk)
    monkeypatch.setattr(Path, "rglob", counting_rglob)
    handles = server_mod.Discovery(root=root).scan()

    assert len(handles) == 8
    assert traversals["n"] <= 2, (
        f"scan traversed the tree {traversals['n']} times for 8 runs; "
        "it must not scale with the number of runs"
    )


def test_scan_still_reads_status_and_evidence_from_the_ledger(tmp_path: Path) -> None:
    """The speed-up must not cost the enrichment it replaced."""
    root = _sweep_root(tmp_path, runs=3)
    handles = server_mod.Discovery(root=root).scan()

    assert {h.status for h in handles} == {"completed"}
    assert all(h.evidence is not None for h in handles)
    assert all(h.evidence.parent.name == "evidence" for h in handles)
    assert {h.instance for h in handles} == {"inst-0", "inst-1", "inst-2"}
    assert {h.arm for h in handles} == {"A3"}
