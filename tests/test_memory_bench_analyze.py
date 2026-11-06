"""The comparison's last stage: it refuses a run that is not whole and one run, and its numbers are worked out by hand here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.memory_bench import analyze
from studies.memory_bench.analyze import check_run, readers_of, report, sign_test

COMMIT, DIGEST = "abc123", "d" * 64
QUESTIONS = ["q1", "q2", "q3", "q4_abs", "d6233ab6"]
LOCOMO = ["conv-1:0", "conv-1:1", "conv-2:0", "conv-2:1"]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(root: Path, view: str, row: str, right: dict[str, bool] | dict[str, float], *, readers: tuple[str, ...] = ("llama", "oss")) -> None:
    budget = int(row.rsplit("-", 1)[1])
    records = root / "records" / view / row
    questions = list(right)
    _write(records / "records.jsonl", [
        {"qid": f"n{q}", "question_id": q, "question_type": "multi-session", "category": 4, "prompt_sha256": f"hash-{row}-{q}",
         "tokens_delivered": 100, "dropped": 2, "headers_missing": 0,
         "scores": ({"session_recall_all@budget": 1.0, "session_recall_any@10": 1.0, "session_recall@budget": 1.0} if not q.endswith("_abs") else {})}
        for q in questions])
    (records / "records.settings.json").write_text(json.dumps({"commit": COMMIT, "code_digest": DIGEST, "part": "dev", "budget": budget}))
    for reader in readers:
        cap, effort = analyze.READER_SETTINGS[reader, view]
        answers = root / f"answers-{reader}" / view / row / "answers.jsonl"
        _write(answers, [{"qid": f"n{q}", "question_id": q, "prompt_sha256": f"hash-{row}-{q}", "hypothesis": "x", "max_tokens": cap, "effort": effort,
                          "study_commit": COMMIT, "code_digest": DIGEST, "finish_reason": "stop"} for q in questions])
        if view == "lme":
            _write(answers.with_name(answers.name + ".eval-results-served"),
                   [{"question_id": q, "autoeval_label": {"label": right[q]}} for q in questions])
        else:
            scored = [{"qid": f"n{q}", "question_id": q, "category": 4, "score": right[q], "audit_flagged": False} for q in questions]
            _write(answers.with_name("scores.jsonl"), scored)
            mean = sum(right.values()) / len(right)
            answers.with_name("scores.summary.json").write_text(json.dumps({
                "scored_1_to_4": len(right), "f1_categories_1_to_4": mean, "f1_without_audit_flags": mean, "per_category": {"4": mean},
                "adversarial": None, "audit_sha256": "e" * 64}))


def _rank(root: Path, view: str, arm: str, questions: list[str], **versions: object) -> None:
    folder = root / "rankings" / view / arm
    _write(folder / "rankings.jsonl", [{"qid": f"n{q}", "versions": {"study_commit": COMMIT, "code_digest": DIGEST, **versions}} for q in questions])
    if arm == "mem0":
        _write(folder / "shard0.replies.jsonl", [{"parsed": True, "reply": "{}"}])


@pytest.fixture
def run(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setitem(analyze.EXPECTED, ("lme", "dev"), len(QUESTIONS))
    monkeypatch.setitem(analyze.EXPECTED, ("locomo", "dev"), len(LOCOMO))
    root = tmp_path / "dev-abc"
    yes = dict.fromkeys(QUESTIONS, True)
    _row(root, "lme", "flat_bm25-8000", yes)
    _row(root, "lme", "taste_git-8000", yes)
    _row(root, "lme", "mem0-8000", {**yes, "q1": False, "q2": False})
    _row(root, "lme", "mempalace_bench_v4-8000", {**yes, "d6233ab6": False})
    _row(root, "lme", "full_llama-31000", yes, readers=("llama",))
    _row(root, "lme", "mem0-2000", yes, readers=("oss",))
    scores = dict.fromkeys(LOCOMO, 0.5)
    _row(root, "locomo", "flat_bm25-8000", scores)
    _row(root, "locomo", "taste_git-8000", scores)
    _row(root, "locomo", "mem0-8000", {**scores, "conv-1:0": 1.0})
    for view, questions in (("lme", QUESTIONS), ("locomo", LOCOMO)):
        _rank(root, view, "mem0", questions, **analyze.MEM0_SETTINGS)
    return root


def _check(root: Path) -> list[str]:
    return check_run(root, part="dev", commit=COMMIT, digest=DIGEST)


def test_a_whole_run_passes_and_its_tables_are_right(run: Path) -> None:
    assert _check(run) == []
    text, data = report(run)
    lme = {entry["row"]: entry for entry in data["lme"]["oss"]}
    assert lme["taste_git-8000"]["accuracy"] == 1.0 and "against_reference" not in lme["taste_git-8000"]
    against = lme["mem0-8000"]["against_reference"]
    assert (against["better"], against["worse"]) == (0, 2) and against["p"] == pytest.approx(0.5)
    assert against["difference"] == pytest.approx(-0.4)
    assert lme["mem0-8000"]["retrieval_by_answer"] == {"right_right": 2, "right_wrong": 2, "wrong_right": 0, "wrong_wrong": 0}
    assert "mem0-8000" in text and "reader llama" in text
    locomo = {entry["row"]: entry for entry in data["locomo"]["llama"]}
    assert locomo["mem0-8000"]["against_reference"]["difference"] == pytest.approx(0.125)
    assert locomo["mem0-8000"]["against_reference"]["conversations"] == 2


def test_hybrid_v4_is_also_scored_without_the_questions_it_was_tuned_on(run: Path) -> None:
    _, data = report(run)
    v4 = next(entry for entry in data["lme"]["oss"] if entry["row"] == "mempalace_bench_v4-8000")
    assert v4["accuracy"] == pytest.approx(0.8)
    assert v4["without_named_questions"] == {"n": 4, "named_here": 1, "accuracy": 1.0}


def test_identical_prompts_sent_twice_give_the_noise_floor(run: Path) -> None:
    for row in ("flat_bm25-8000", "taste_git-8000"):
        for name in ("answers.jsonl",):
            path = run / "answers-oss" / "lme" / row / name
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            _write(path, [{**answer, "prompt_sha256": "same", "hypothesis": row if answer["question_id"] == "q1" else "x"} for answer in rows])
    _, data = report(run)
    assert data["noise"]["oss"]["lme"] == {"questions": 5, "same_prompt": 5, "same_answer_given_same_prompt": 4, "verdicts_differ": 0}


def test_each_row_goes_to_the_readers_the_protocol_names() -> None:
    assert readers_of("mem0-8000") == ("llama", "oss")
    assert readers_of("mem0-16000") == ("oss",)
    assert readers_of("full_llama-31000") == ("llama",)
    assert readers_of("oracle_oss-126000") == ("oss",)


def _rewrite(path: Path, change) -> None:
    _write(path, [change(json.loads(line)) for line in path.read_text().splitlines()])


def test_a_missing_answer_is_refused(run: Path) -> None:
    path = run / "answers-oss" / "lme" / "mem0-8000" / "answers.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    assert any("mem0-8000 oss: 1 questions have no answer" in problem for problem in _check(run))


def test_an_answer_to_a_prompt_no_longer_on_record_is_refused(run: Path) -> None:
    _rewrite(run / "answers-llama" / "lme" / "mem0-8000" / "answers.jsonl", lambda row: {**row, "prompt_sha256": "old"})
    assert any("5 answers are to prompts no longer on record" in problem for problem in _check(run))


def test_a_missing_verdict_is_refused(run: Path) -> None:
    path = run / "answers-oss" / "lme" / "mem0-8000" / "answers.jsonl.eval-results-served"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    assert any("1 answers have no verdict" in problem for problem in _check(run))


def test_a_row_from_another_commit_or_copy_of_the_code_is_refused(run: Path) -> None:
    path = run / "records" / "lme" / "mem0-8000" / "records.settings.json"
    settings = json.loads(path.read_text())
    path.write_text(json.dumps({**settings, "commit": "other", "code_digest": "other"}))
    problems = _check(run)
    assert any("commit is 'other'" in problem for problem in problems) and any("code_digest is 'other'" in problem for problem in problems)


def test_a_budget_that_is_not_the_rows_name_is_refused(run: Path) -> None:
    path = run / "records" / "lme" / "mem0-8000" / "records.settings.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "budget": 4000}))
    assert any("budget is 4000" in problem for problem in _check(run))


def test_too_few_records_are_refused(run: Path) -> None:
    path = run / "records" / "lme" / "mem0-8000" / "records.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    assert any("4 records, not 5" in problem for problem in _check(run))


def test_lost_headers_are_refused(run: Path) -> None:
    _rewrite(run / "records" / "lme" / "mem0-8000" / "records.jsonl", lambda row: {**row, "headers_missing": 1})
    assert any("headers were lost" in problem for problem in _check(run))


def test_a_reader_on_another_cap_is_refused(run: Path) -> None:
    _rewrite(run / "answers-oss" / "lme" / "mem0-8000" / "answers.jsonl", lambda row: {**row, "max_tokens": 800})
    assert any("max_tokens" in problem for problem in _check(run))


def test_mem0_on_other_settings_is_refused(run: Path) -> None:
    _rewrite(run / "rankings" / "lme" / "mem0" / "rankings.jsonl", lambda row: {**row, "versions": {**row["versions"], "top_k": 200}})
    assert any("mem0: top_k" in problem for problem in _check(run))


def test_mem0_with_too_few_replies_parsed_is_refused(run: Path) -> None:
    _write(run / "rankings" / "lme" / "mem0" / "shard0.replies.jsonl", [{"parsed": False, "reply": "null"}])
    assert any("0 of 1 replies parsed" in problem for problem in _check(run))


def test_a_locomo_row_scored_with_no_audit_file_is_refused(run: Path) -> None:
    path = run / "answers-oss" / "locomo" / "mem0-8000" / "scores.summary.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "audit_sha256": None}))
    assert any("no audit file" in problem for problem in _check(run))


def test_the_sign_test_is_exact() -> None:
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 0) == pytest.approx(2 / 32)


def test_the_command_line_writes_no_tables_for_a_broken_run(run: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "report.md"
    arguments = ["--run", str(run), "--part", "dev", "--commit", "not-the-commit", "--digest", DIGEST, "--out", str(out)]
    assert analyze.main(arguments) == 1
    assert not out.exists() and "problems" in capsys.readouterr().out
    assert analyze.main(["--run", str(run), "--part", "dev", "--commit", COMMIT, "--digest", DIGEST, "--out", str(out)]) == 0
    assert out.exists() and out.with_suffix(".json").exists()
