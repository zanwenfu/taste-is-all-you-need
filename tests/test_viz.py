"""The interactive report.

The dashboard renders a fixed picture; this renders something you can
interrogate. So the tests are about whether the *data behind the clicks* is
right — an observation that says "nothing changed here" when something did is
worse than no report at all — and about the page still building when a run
went badly, which is exactly when someone opens it.

Hermetic: real git, no Docker, no API.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from taste import viz
from taste.viz import DIFF_LINE_CAP, RunPayload, build_payload, render_index, render_run


def _payload_from(html: str) -> dict:
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert match, "the page must embed its data"
    return json.loads(match.group(1))


@pytest.fixture
def run_workspace(tmp_path: Path) -> Path:
    """A real measured run: two steps, the second breaks a passing check."""
    from taste.benchmarks import swebench
    from taste.benchmarks.swebench_run import make_execute, make_prepare, make_score
    from taste.cores import Plan, Step, Verification, WorkerResult
    from taste.evalrun import run_sweep
    from taste.replay import SuiteProbe

    src = tmp_path / "source" / "toy__toy-1"
    src.mkdir(parents=True)
    (src / "lib.py").write_text("def rate():\n    return 1\n")

    instance = swebench.SWEInstance(
        instance_id="toy__toy-1", repo="pytest-dev/pytest", base_commit="0" * 40,
        problem_statement="rate() must stay 1.", test_patch="", version="7.0",
        fail_to_pass=(), pass_to_pass=("t::rate",),
    )

    def suite(_i):
        return SuiteProbe(
            name="p", command=f'{sys.executable} -c "import lib; assert lib.rate() == 1"',
            members=("t::rate",), timeout=30,
        )

    def scripted(_cell, ctx):
        ws = ctx.workspace
        plan = Plan(task="toy", steps=[
            Step(id="step-01", description="add helper",
                 verification=Verification(kind="shell", command="test -f helper.py")),
            Step(id="step-02", description="break it",
                 verification=Verification(kind="shell", command="test -f note.txt")),
        ])

        def worker(step, _p):
            if step.id == "step-01":
                (ws / "helper.py").write_text("# helper\n")
                return WorkerResult("helper", 2, "end_turn")
            (ws / "note.txt").write_text("note\n")
            (ws / "lib.py").write_text("def rate():\n    return 2\n")
            return WorkerResult("note", 3, "end_turn")

        return {"plan_override": plan, "worker_override": worker}

    ledger = tmp_path / "ledger"
    report = run_sweep(
        tasks=["toy__toy-1"], arms=["A0"], trials=1, ledger_dir=ledger,
        prepare=make_prepare(instances={"toy__toy-1": instance}, root=tmp_path / "runs",
                             source_root=tmp_path / "source", provider=None),
        execute=make_execute(run_overrides=scripted),
        score=make_score(ledger_dir=ledger, suite_factory=suite),
    )
    cell = report.results[0]
    ws = Path(cell.workspace)
    (ws / ".taste" / "evidence.json").write_text(Path(cell.report_path).read_text())
    return ws


# ------------------------------------------------------------------ payload


def test_an_observation_records_what_changed_there(run_workspace: Path) -> None:
    """The whole point of clicking one. A diff that is silently empty makes the
    report actively misleading."""
    payload = build_payload(run_workspace)
    edits = [o for o in payload.observations if "lib.py" in o["files"]]
    assert edits, "no observation recorded the edit to lib.py"
    assert "return 2" in edits[-1]["diff"]
    assert "-    return 1" in edits[-1]["diff"]


def test_the_broken_test_is_named_at_the_observation_it_broke(run_workspace: Path) -> None:
    payload = build_payload(run_workspace)
    broken = [o for o in payload.observations if o["broken"]]
    assert broken, "the regression was never attached to an observation"
    assert broken[0]["broken"] == ["t::rate"]


def test_the_monitor_verdict_travels_with_the_observation(run_workspace: Path) -> None:
    payload = build_payload(run_workspace)
    verdicts = [o["monitor"] for o in payload.observations if o["monitor"] is not None]
    assert verdicts, "no verdict was placed on the timeline"


def test_steps_carry_their_checkpoint_cards(run_workspace: Path) -> None:
    payload = build_payload(run_workspace)
    assert payload.steps
    assert all("cards" in s for s in payload.steps)


def test_diffs_can_be_skipped_for_a_much_smaller_file(run_workspace: Path) -> None:
    """A real repository observation can carry thousands of lines."""
    with_diffs = build_payload(run_workspace, with_diffs=True)
    without = build_payload(run_workspace, with_diffs=False)
    assert any(o["diff"] for o in with_diffs.observations)
    assert not any(o["diff"] for o in without.observations)


def test_a_long_diff_is_capped_and_says_so(tmp_path: Path) -> None:
    """A silently shortened diff is worse than no diff."""
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "big.py").write_text("x = 0\n")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.co",
                    "commit", "-qm", "base"], cwd=ws, check=True)
    (ws / "big.py").write_text("\n".join(f"x = {i}" for i in range(DIFF_LINE_CAP * 2)))
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.co",
                    "commit", "-qm", "big"], cwd=ws, check=True)

    text, truncated = viz._diff_between(ws, "HEAD~1", "HEAD")
    assert truncated is True
    assert len(text.splitlines()) <= DIFF_LINE_CAP


def test_a_missing_object_degrades_instead_of_raising(tmp_path: Path) -> None:
    """gc'd shadow chain, moved workspace — the report must still build."""
    ws = tmp_path / "repo"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    text, truncated = viz._diff_between(ws, "deadbeef", "cafebabe")
    assert "unavailable" in text
    assert truncated is False


# ------------------------------------------------------------------ page


def test_the_page_embeds_its_data_and_needs_no_network(run_workspace: Path) -> None:
    page = render_run(build_payload(run_workspace))
    assert _payload_from(page)["observations"]
    for forbidden in ("http://", "https://", "<script src", "@import"):
        assert forbidden not in page, f"page reaches outside itself: {forbidden}"


def test_the_page_declares_no_document_scaffolding(run_workspace: Path) -> None:
    """It is embedded, so it must not bring its own html/body."""
    page = render_run(build_payload(run_workspace))
    assert page.lstrip().startswith("<title>")
    for tag in ("<!DOCTYPE", "<html", "<body"):
        assert tag not in page


def test_a_closing_tag_inside_the_data_cannot_break_out(tmp_path: Path) -> None:
    """Task text is untrusted and lands inside a <script> block."""
    payload = RunPayload(session="s", task="</script><img src=x onerror=alert(1)>")
    page = render_run(payload)
    assert "</script><img" not in page
    assert "<\\/script>" in page


def test_the_report_states_its_own_limits(tmp_path: Path) -> None:
    """An unmeasured run must not read as a clean one."""
    ws = tmp_path / "bare"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    payload = build_payload(ws)
    joined = " ".join(payload.notes)
    assert "not measured" in joined
    assert "unknown rather than absent" in joined


# ------------------------------------------------------------------ index


def test_the_index_summarises_every_cell(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    (ledger / "evidence").mkdir(parents=True)
    (ledger / "a__A0__t1.json").write_text(json.dumps({
        "task": "inst-a", "arm": "A0", "trial": 1, "status": "completed",
        "billed_usd": 0.5, "steps_passed": 2, "steps_total": 2, "report_path": "r.html",
    }))
    (ledger / "evidence" / "a__A0__t1.json").write_text(json.dumps({
        "observations": 4, "episodes": [{"probe": "t"}], "silence": {"silent_attributed": 1},
    }))
    out = viz.write_index(ledger)
    rows = _payload_from(out.read_text())["rows"]

    assert len(rows) == 1
    assert rows[0]["silent"] == 1 and rows[0]["episodes"] == 1
    assert rows[0]["observations"] == 4


def test_the_index_survives_a_corrupt_ledger_entry(tmp_path: Path) -> None:
    """A killed sweep leaves a half-written file; the other cells still matter."""
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "good.json").write_text(json.dumps({"task": "t", "arm": "A0", "trial": 1}))
    (ledger / "bad.json").write_text('{"task": "t"')
    rows = _payload_from(viz.write_index(ledger).read_text())["rows"]
    assert len(rows) == 1


def test_an_empty_index_renders(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    assert "Sweep" in render_index([])
    assert _payload_from(viz.write_index(ledger).read_text())["rows"] == []
