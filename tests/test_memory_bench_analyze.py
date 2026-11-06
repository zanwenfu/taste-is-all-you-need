"""The comparison's last stage: it refuses a run that is not whole and one run, and its numbers are worked out by hand here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.memory_bench import analyze
from studies.memory_bench.analyze import check_run, kappa, readers_of, report, sign_test
from studies.memory_bench.core import DATA_SHA256

COMMIT, DIGEST = "abc123", "d" * 64
QUESTIONS = ["q1", "q2", "q3", "q4_abs", "d6233ab6"]
LOCOMO = ["conv-1:0", "conv-1:1", "conv-2:0", "conv-2:1"]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _settings(root: Path, view: str, row: str) -> dict:
    wanted = analyze.made_with(view, row)
    name = row.rsplit("-", 1)[0]
    settings = {"commit": COMMIT, "code_digest": DIGEST, "part": "dev", "budget": int(row.rsplit("-", 1)[1]), "benchmark": view, "system": wanted["system"],
                "charge": wanted["charge"], "tokenizer": f"/models/{wanted['tokenizer']}"}
    if wanted["system"] == "precomputed":
        settings["rankings_sha256"] = analyze._sha256(root / "rankings" / view / name / "rankings.jsonl")
    if name == "taste_git_headers":
        settings["headers_sha256"] = analyze._sha256(root / "headers" / f"{view}.jsonl")
    return settings


def _judge_settings(**more: object) -> dict:
    return {**analyze.JUDGE_SETTINGS, "commit": COMMIT, "code_digest": DIGEST, "reference_sha256": DATA_SHA256["lme"], **more}


def _row(root: Path, view: str, row: str, right: dict[str, bool] | dict[str, float], *, readers: tuple[str, ...] | None = None) -> None:
    records = root / "records" / view / row
    questions = list(right)
    _write(records / "records.jsonl", [
        {"qid": f"n{q}", "question_id": q, "question_type": "multi-session", "category": 4, "prompt_sha256": f"hash-{row}-{q}",
         "tokens_delivered": 100, "dropped": 2, "headers_missing": 0,
         "scores": ({"session_recall_all@budget": 1.0, "session_recall_any@10": 1.0, "session_recall@budget": 1.0} if not q.endswith("_abs") else {})}
        for q in questions])
    (records / "records.settings.json").write_text(json.dumps(_settings(root, view, row)))
    (records / "records.summary.json").write_text("{}")
    for reader in readers or readers_of(row):
        cap, effort = analyze.reader_cap(reader, view, row)
        answers = root / f"answers-{reader}" / view / row / "answers.jsonl"
        _write(answers, [{"qid": f"n{q}", "question_id": q, "prompt_sha256": f"hash-{row}-{q}", "hypothesis": "x", "max_tokens": cap, "effort": effort,
                          "study_commit": COMMIT, "code_digest": DIGEST, "finish_reason": "stop"} for q in questions])
        if view == "lme":
            verdicts = answers.with_name(answers.name + ".eval-results-served")
            _write(verdicts, [{"question_id": q, "prompt_sha256": f"hash-{row}-{q}", "hypothesis": "x",
                               "autoeval_label": {"label": right[q], "model": analyze.OSS_MODEL, "official_rule_label": right[q]}} for q in questions])
            Path(f"{verdicts}.settings.json").write_text(json.dumps(_judge_settings(answers_sha256=analyze._sha256(answers))))
        else:
            scored = [{"qid": f"n{q}", "question_id": q, "category": 4, "score": right[q], "audit_flagged": False} for q in questions]
            _write(answers.with_name("scores.jsonl"), scored)
            mean = sum(right.values()) / len(right)
            answers.with_name("scores.summary.json").write_text(json.dumps({
                "scored_1_to_4": len(right), "f1_categories_1_to_4": mean, "f1_without_audit_flags": mean, "per_category": {"4": mean},
                "adversarial": None, "audit_sha256": "e" * 64, "answers_sha256": analyze._sha256(answers), "commit": COMMIT, "code_digest": DIGEST}))
    if view == "lme" and row.endswith("-8000") and row != "none-8000":
        enough = root / "enough" / "lme" / row / "verdicts.jsonl"
        _write(enough, [{"question_id": q, "prompt_sha256": f"hash-{row}-{q}", "enough": True} for q in questions])
        enough.with_suffix(".settings.json").write_text(json.dumps(_judge_settings(max_tokens=2048)))


def _rank(root: Path, view: str, arm: str, questions: list[str], **versions: object) -> None:
    folder = root / "rankings" / view / arm
    _write(folder / "rankings.jsonl", [{"qid": f"n{q}", "versions": {"study_commit": COMMIT, "code_digest": DIGEST, **versions}} for q in questions])
    if arm == "mem0":
        _write(folder / "shard0.replies.jsonl", [{"parsed": True, "reply": "{}"}])


def _headers(root: Path, view: str, **more: object) -> None:
    settings = {**analyze.HEADER_SETTINGS, "prompt_sha256": "p" * 64, "study_commit": COMMIT, "code_digest": DIGEST, **more}
    _write(root / "headers" / f"{view}.jsonl", [{"key": "k", "ok": True, **settings}])


@pytest.fixture
def run(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setitem(analyze.EXPECTED, ("lme", "dev"), len(QUESTIONS))
    monkeypatch.setitem(analyze.EXPECTED, ("locomo", "dev"), len(LOCOMO))
    root = tmp_path / "dev-abc"
    yes = dict.fromkeys(QUESTIONS, True)
    scores = dict.fromkeys(LOCOMO, 0.5)
    for view, questions, right in (("lme", QUESTIONS, yes), ("locomo", LOCOMO, scores)):
        _headers(root, view)
        for arm in analyze.ARMS[view]:
            _rank(root, view, arm, questions, **(analyze.MEM0_SETTINGS if arm == "mem0" else {}))
        for row in sorted(analyze.required_rows(view)):
            _row(root, view, row, right)
    _row(root, "lme", "mem0-8000", {**yes, "q1": False, "q2": False})
    _row(root, "lme", "mempalace_bench_v4-8000", {**yes, "d6233ab6": False})
    _row(root, "locomo", "mem0-8000", {**scores, "conv-1:0": 1.0})
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
    assert (v4["without_named_questions"]["n"], v4["without_named_questions"]["named_here"], v4["without_named_questions"]["accuracy"]) == (4, 1, 1.0)
    assert v4["without_named_questions"]["session_recall_any@10"] == 1.0, "what it was tuned on was retrieval, so retrieval is read without them too"


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
    assert readers_of("oracle_oss-118000") == ("oss",)


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


def _drop(path: Path) -> None:
    import shutil

    shutil.rmtree(path) if path.is_dir() else path.unlink()


def _with(path: Path, **more: object) -> None:
    path.write_text(json.dumps({**json.loads(path.read_text()), **more}))


def _answers(run: Path, row: str = "mem0-8000", reader: str = "oss", view: str = "lme") -> Path:
    return run / f"answers-{reader}" / view / row / "answers.jsonl"


@pytest.mark.parametrize(("break_it", "says"), [
    (lambda run: _drop(run / "records" / "lme" / "mempalace_bench_raw-8000"), "lme: no mempalace_bench_raw-8000 row"),
    (lambda run: _drop(run / "records" / "lme" / "flat_bm25-8000"), "lme: no flat_bm25-8000 row"),
    (lambda run: _drop(run / "rankings" / "locomo" / "mem0"), "locomo mem0: no rankings"),
    (lambda run: (run / "records" / "lme" / "made_up-8000").mkdir(), "made_up-8000 is not a row of the protocol"),
    (lambda run: _drop(run / "records" / "lme" / "mem0-8000" / "records.summary.json"), "records.summary.json is missing"),
    (lambda run: _rewrite(_answers(run), lambda row: {**row, "hypothesis": ""}), "mem0-8000 oss: 5 answers are empty"),
    (lambda run: _rewrite(_answers(run), lambda row: {**row, "finish_reason": "length"}), "mem0-8000 oss: 5 answers were cut off"),
    (lambda run: _rewrite(Path(f"{_answers(run)}.eval-results-served"), lambda row: {**row, "hypothesis": "older"}), "5 verdicts are of another answer"),
    (lambda run: _rewrite(Path(f"{_answers(run)}.eval-results-served"), lambda row: {**row, "autoeval_label": {**row["autoeval_label"], "label": None}}), "5 verdicts could not be read"),
    (lambda run: _with(Path(f"{_answers(run)}.eval-results-served.settings.json"), model="another"), "judge's model is 'another'"),
    (lambda run: _with(Path(f"{_answers(run)}.eval-results-served.settings.json"), reference_sha256="oracle"), "judge's reference_sha256 is 'oracle'"),
    (lambda run: _with(Path(f"{_answers(run)}.eval-results-served.settings.json"), temperature=1), "judge's temperature is 1"),
    (lambda run: _write(run / "enough" / "lme" / "mem0-8000" / "verdicts.jsonl", [{"question_id": "q1", "prompt_sha256": "hash-mem0-8000-q1", "enough": True}]), "enough: 4 questions have no verdict"),
    (lambda run: _drop(run / "enough" / "lme" / "mem0-8000"), "enough: no verdicts"),
    (lambda run: _with(run / "records" / "lme" / "mem0-8000" / "records.settings.json", charge="prompt"), "charge is 'prompt'"),
    (lambda run: _with(run / "records" / "lme" / "mem0-8000" / "records.settings.json", tokenizer="/models/gpt-oss-120b"), "tokenizer is"),
    (lambda run: _with(run / "records" / "lme" / "mem0-8000" / "records.settings.json", system="flat_bm25"), "system is 'flat_bm25'"),
    (lambda run: _rewrite(run / "rankings" / "lme" / "mem0" / "rankings.jsonl", lambda row: {**row, "changed": True}), "lme mem0-8000: its rankings are not the ones the run holds"),
    (lambda run: _headers(run, "lme", changed=True), "lme taste_git_headers-8000: its headers are not the ones the run holds"),
    (lambda run: _rewrite(run / "rankings" / "lme" / "mem0" / "rankings.jsonl", lambda row: {**row, "versions": {**row["versions"], "embedder": "another"}}), "mem0: embedder"),
    (lambda run: _rewrite(run / "rankings" / "lme" / "mem0" / "rankings.jsonl", lambda row: {**row, "versions": {**row["versions"], "mem0ai": "2.0"}}), "mem0: mem0ai"),
    (lambda run: _rewrite(run / "rankings" / "lme" / "mem0" / "rankings.jsonl", lambda row: {**row, "qid": "same"}), "a question was ranked twice"),
    (lambda run: _write(run / "rankings" / "lme" / "flat_dense" / "rankings.jsonl", [{"qid": "n1", "versions": {"study_commit": COMMIT, "code_digest": DIGEST}}]), "flat_dense: 1 rankings, not 5"),
    (lambda run: _with(_answers(run, view="locomo").with_name("scores.summary.json"), answers_sha256="older"), "scored from other answers"),
    (lambda run: _with(_answers(run, view="locomo").with_name("scores.summary.json"), commit="older"), "scored by commit 'older'"),
    (lambda run: _headers(run, "lme", max_tokens=100), "lme headers: max_tokens"),
    (lambda run: _headers(run, "lme", study_commit="older"), "lme headers: study_commit"),
    (lambda run: _drop(run / "headers" / "lme.jsonl"), "lme headers: no file"),
])
def test_a_run_that_is_not_whole_is_refused_however_it_broke(run: Path, break_it, says: str) -> None:
    assert _check(run) == []
    break_it(run)
    assert any(says in problem for problem in _check(run)), _check(run)


def test_a_few_empty_answers_are_a_count_and_many_stop_the_row(run: Path, monkeypatch) -> None:
    monkeypatch.setattr(analyze, "MOST_EMPTY", 0.25)
    rows = [json.loads(line) for line in _answers(run).read_text().splitlines()]
    _write(_answers(run), [{**rows[0], "hypothesis": ""}, *rows[1:]])
    judged = Path(f"{_answers(run)}.eval-results-served")
    _rewrite(judged, lambda row: {**row, "hypothesis": "" if row["question_id"] == rows[0]["question_id"] else row["hypothesis"]})
    _with(Path(f"{judged}.settings.json"), answers_sha256=analyze._sha256(_answers(run)))
    assert _check(run) == []
    _, data = report(run)
    assert next(e for e in data["lme"]["oss"] if e["row"] == "mem0-8000")["empty"] == 1


def test_a_full_context_answer_gets_the_higher_cap_and_no_other_row_does() -> None:
    assert analyze.reader_cap("oss", "lme", f"full_oss-{analyze.OSS_WINDOW}") == (analyze.FULL_CONTEXT_CAP, "low")
    assert analyze.reader_cap("oss", "locomo", f"full_oss-{analyze.OSS_WINDOW}") == (analyze.FULL_CONTEXT_CAP, "low")
    assert analyze.reader_cap("oss", "lme", "mem0-8000") == (4096, "low") and analyze.reader_cap("oss", "locomo", "mem0-8000") == (4096, "low")
    assert analyze.reader_cap("llama", "lme", "full_llama-31000") == (800, None) and analyze.reader_cap("llama", "locomo", "mem0-8000") == (50, None)
    assert analyze.OSS_WINDOW + analyze.FULL_CONTEXT_CAP < 131072


def test_a_row_of_the_sweep_is_read_against_taste_at_its_own_budget(run: Path) -> None:
    yes = dict.fromkeys(QUESTIONS, True)
    _row(run, "lme", "taste_git-2000", {**yes, "q1": False, "q2": False, "q3": False})
    _row(run, "lme", "mem0-2000", {**yes, "q1": False})
    _, data = report(run)
    against = next(e for e in data["lme"]["oss"] if e["row"] == "mem0-2000")["against_reference"]
    assert against["difference"] == pytest.approx(0.4), "two better than Taste at 2,000; at 8,000 it would read one worse"
    assert (against["better"], against["worse"]) == (2, 0)


def test_two_judges_are_compared_on_the_answers_both_read(run: Path) -> None:
    second = Path(f"{_answers(run)}.eval-results-gpt-5.4")
    second.write_text(Path(f"{_answers(run)}.eval-results-served").read_text())
    _, data = report(run)
    assert data["judges"]["oss"] == {"judge": "gpt-5.4", "answers": 5, "agree": 5, "kappa": 1.0}
    assert kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)


def test_the_ledger_counts_tokens_and_gpu_hours_stage_by_stage(run: Path) -> None:
    job = run / "answers-oss" / "job-1"
    job.mkdir()
    (job / "vllm-metrics.txt").write_text('vllm:prompt_tokens_total{engine="0"} 100.0\nvllm:generation_tokens_total{engine="0"} 20.0\n'
                                          'vllm:prefix_cache_hits_total{engine="0"} 60.0\nvllm:request_success_total{finished_reason="stop"} 3.0\n')
    (job / "gpu.csv").write_text("timestamp, index\n" + "x, 0\n" * 120)
    _, data = report(run)
    assert data["ledger"]["answers-oss"] == {"prompt_tokens": 100.0, "generated_tokens": 20.0, "cached_tokens": 60.0, "calls": 3.0, "gpu_hours": 2.0}


def test_a_later_commit_may_carry_on_a_run_only_if_the_ingest_it_stands_on_is_unchanged(run: Path, tmp_path: Path, monkeypatch) -> None:
    from studies.memory_ledger import code_digest, ingest_digest

    def tree(name: str, adapter: str) -> Path:
        base = tmp_path / name
        for path, text in (("taste/a.py", "a"), ("studies/memory_bench/adapters/mem0_arm.py", adapter), ("studies/memory_bench/headers.py", "h")):
            (base / path).parent.mkdir(parents=True, exist_ok=True)
            (base / path).write_text(text)
        return base

    earlier = tree("earlier", "the same")
    monkeypatch.setattr("studies.memory_ledger.ingest_digest", lambda root=None: ingest_digest(root if root else earlier))
    digest = code_digest(earlier)
    assert analyze._check_ingest_code("old", digest, str(earlier)) == []
    assert any("is not the code of" in problem for problem in analyze._check_ingest_code("old", "another", str(earlier)))
    changed = tree("changed", "another")
    assert any("must be run again" in problem for problem in analyze._check_ingest_code("old", code_digest(changed), str(changed)))
    assert analyze._check_ingest_code("old", None, None) != []
    # rankings stamped by the earlier commit, the later stages by this one
    for view in ("lme", "locomo"):
        for arm in analyze.ARMS[view]:
            _rewrite(run / "rankings" / view / arm / "rankings.jsonl", lambda row: {**row, "versions": {**row["versions"], "study_commit": "old", "code_digest": "old-digest"}})
        _rewrite(run / "headers" / f"{view}.jsonl", lambda row: {**row, "study_commit": "old", "code_digest": "old-digest"})
        for row in analyze.required_rows(view):
            path = run / "records" / view / row / "records.settings.json"
            wanted = _settings(run, view, row)
            path.write_text(json.dumps(wanted))
    assert any("study_commit" in problem for problem in _check(run)), "not said to be from an earlier commit"
    assert check_run(run, part="dev", commit=COMMIT, digest=DIGEST, ingest_commit="old", ingest_digest="old-digest") == []


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
