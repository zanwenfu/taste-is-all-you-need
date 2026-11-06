"""The paid second judge: which rows, one limit over the whole job, small batches, and no call on anything but the real test."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from studies.context_rot.lme_judge import SpendLimit
from studies.memory_bench import analyze, paid_judge


class _Judge:
    """Stands in for the paid model: every answer costs a cent."""

    model, effort, max_tokens = "stand-in", "low", 512

    def __init__(self, text: str = "yes", cost: float = 0.01) -> None:
        self.asked, self.text, self.price = 0, text, cost

    def ask(self, prompt: str) -> types.SimpleNamespace:
        self.asked += 1
        return types.SimpleNamespace(text=self.text, input_tokens=1, output_tokens=1, stop_reason="end_turn", cost=lambda model: self.price)


def _copy(root: Path, count: int = 10, rows: tuple[tuple[str, str], ...] = (("oss", "mem0-8000"),), free: bool = True) -> Path:
    for reader, row in rows:
        folder = root / f"answers-{reader}" / "lme" / row
        folder.mkdir(parents=True)
        (folder / "answers.jsonl").write_text("".join(json.dumps({"question_id": f"q{n}", "hypothesis": "x"}) + "\n" for n in range(count)))
        if free:
            (folder / "answers.jsonl.eval-results-served").write_text("".join(json.dumps({"question_id": f"q{n}", "autoeval_label": {"label": n % 2 == 0}}) + "\n" for n in range(count)))
    reference = root / "reference.json"
    reference.write_text(json.dumps([{"question_id": f"q{n}", "question_type": "multi-session", "question": "Q?", "answer": "A"} for n in range(count)]))
    return reference


def test_it_judges_the_rows_the_main_table_reads_for_both_readers_and_no_sweep() -> None:
    rows = analyze.second_judge_rows()
    assert len(rows) == 22 and len(rows) * 450 == 9900
    assert sum(reader == "oss" for reader, _ in rows) == 11 and sum(reader == "llama" for reader, _ in rows) == 11
    assert ("oss", f"full_oss-{analyze.OSS_WINDOW}") in rows and ("llama", "full_llama-31000") in rows and ("oss", f"oracle_oss-{analyze.OSS_WINDOW}") in rows
    assert ("oss", "full_llama-31000") not in rows and ("llama", f"oracle_oss-{analyze.OSS_WINDOW}") not in rows
    assert not any(row.endswith(("-2000", "-4000", "-16000")) for _, row in rows)
    assert not any(row == "none-8000" for _, row in rows)  # no memory has nothing to judge against a memory's: the free judge reads it
    assert ("llama", "taste_git-8000") in rows and ("oss", "mempalace_product-8000") in rows


def test_small_batches_ask_only_what_has_no_verdict(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    judge = _Judge()
    first = paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", judge, first=3, allowed=None)
    assert judge.asked == 3 and first["judged"] == 3 and first["of"] == 10 and first["spent_total"] == pytest.approx(0.03)
    second = paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", judge, first=2, allowed=None)
    assert judge.asked == 5 and second["judged"] == 5
    third = paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", judge, allowed=None)
    assert judge.asked == 10 and third["judged"] == 10 and third["unparsed"] == 0


def test_one_limit_holds_over_the_whole_job_and_what_was_judged_is_kept(tmp_path: Path) -> None:
    rows = (("oss", "mem0-8000"), ("oss", "taste_git-8000"))
    reference = _copy(tmp_path, rows=rows)
    paid_judge.run_row(tmp_path, reference, *rows[0], _Judge(), allowed=None, workers=1)
    judge = _Judge()
    with pytest.raises(SpendLimit):
        paid_judge.run_row(tmp_path, reference, *rows[1], judge, allowed=None, workers=1, stop_at_usd=0.13)
    assert judge.asked == 3
    assert paid_judge.spent(tmp_path) == pytest.approx(0.13)
    assert len((tmp_path / "answers-oss" / "lme" / "taste_git-8000" / "answers.jsonl.eval-results-gpt-5.4").read_text().splitlines()) == 3


def test_it_says_so_the_moment_the_spend_passes_the_estimate_and_projects_the_whole_job(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    under = paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), first=2, allowed=None, estimate_usd=0.05)
    assert under["past_estimate"] is False
    assert under["projected_total"] == pytest.approx(0.01 * paid_judge.VERDICTS)
    over = paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), first=4, allowed=None, estimate_usd=0.05)
    assert over["past_estimate"] is True


def test_a_row_outside_the_list_is_refused_before_any_call(tmp_path: Path) -> None:
    reference = _copy(tmp_path, rows=(("oss", "mem0-2000"), ("llama", f"full_oss-{analyze.OSS_WINDOW}")))
    judge = _Judge()
    with pytest.raises(ValueError, match="not a row"):
        paid_judge.run_row(tmp_path, reference, "oss", "mem0-2000", judge, allowed=None)
    with pytest.raises(ValueError, match="not a row"):
        paid_judge.run_row(tmp_path, reference, "llama", f"full_oss-{analyze.OSS_WINDOW}", judge, allowed=None)
    assert judge.asked == 0


def test_the_users_yes_covers_only_the_real_test_so_practice_answers_are_refused_before_any_call(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    judge = _Judge()
    with pytest.raises(ValueError, match="not the real test"):
        paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", judge, allowed={"q0", "q1"})
    assert judge.asked == 0
    assert paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", judge, allowed={f"q{n}" for n in range(10)}, first=1)["judged"] == 1


def test_a_reply_that_is_neither_yes_nor_no_is_kept_and_counted(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    report = paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(text="perhaps"), first=4, allowed=None)
    assert report["unparsed"] == 4 and report["judged"] == 4


def test_each_report_compares_with_the_free_judge_and_the_analysis_reads_the_same_files(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    report = paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), allowed=None)
    assert report["free_agrees"] == 5 and report["free_read"] == 10
    agreed = analyze.judges_agree(tmp_path, "oss", "gpt-5.4")
    assert agreed["answers"] == 10 and agreed["agree"] == 5


def test_the_settings_go_beside_the_verdicts_and_hold_what_was_read(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), first=1, allowed=None, commit="c1")
    folder = tmp_path / "answers-oss" / "lme" / "mem0-8000"
    settings = json.loads((folder / "answers.jsonl.eval-results-gpt-5.4.settings.json").read_text())
    assert settings["model"] == "stand-in" and settings["max_tokens"] == 512 and settings["effort"] == "low" and settings["commit"] == "c1"
    assert settings["answers_sha256"] == analyze._sha256(folder / "answers.jsonl") and settings["reference_sha256"] == analyze._sha256(reference)


def test_the_reference_holds_what_a_judge_reads_and_no_history(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    entries = [{"question_id": f"q{n}", "question_type": "multi-session", "question": "Q?", "answer": "A", "question_date": "2023/01/01", "haystack_sessions": [["long"]]} for n in range(3)]
    data.write_text(json.dumps(entries))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["q0"], "held_out": ["q1", "q2"]}))
    out = tmp_path / "reference.json"
    sha = analyze._sha256(data)
    assert paid_judge.cut_reference(data, split, "held_out", out, sha256=sha, now=paid_judge.OPENS_AT) == 2
    rows = json.loads(out.read_text())
    assert [row["question_id"] for row in rows] == ["q1", "q2"]
    assert set(rows[0]) == {"question_id", "question_type", "question", "answer", "question_date"}
    assert json.loads(Path(f"{out}.source.json").read_text()) == {"part": "held_out", "data_sha256": sha}


def test_the_reference_refuses_another_data_file_and_the_real_tests_gold_before_it_opens(tmp_path: Path) -> None:
    from datetime import timedelta

    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"question_id": "q1", "question_type": "multi-session", "question": "Q?", "answer": "A", "question_date": "d"}]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["q1"], "held_out": ["q1"]}))
    out = tmp_path / "reference.json"
    with pytest.raises(SystemExit, match="not the benchmark's file"):
        paid_judge.cut_reference(data, split, "held_out", out, sha256="other", now=paid_judge.OPENS_AT)
    with pytest.raises(SystemExit, match="opens at"):
        paid_judge.cut_reference(data, split, "held_out", out, sha256=analyze._sha256(data), now=paid_judge.OPENS_AT - timedelta(seconds=1))
    assert not out.exists()
    assert paid_judge.cut_reference(data, split, "dev", out, sha256=analyze._sha256(data), now=paid_judge.OPENS_AT - timedelta(days=1)) == 1


def test_a_stand_ins_verdicts_never_count_as_the_paid_judges(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    folder = tmp_path / "answers-oss" / "lme" / "mem0-8000"
    paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), allowed=None, first=4, short_name=paid_judge.STAND_IN)
    assert (folder / "answers.jsonl.eval-results-stand-in").exists() and not (folder / "answers.jsonl.eval-results-gpt-5.4").exists()
    assert paid_judge.spent(tmp_path) == 0.0
    real = _Judge()
    report = paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", real, allowed=None)
    assert real.asked == 10 and report["judged"] == 10 and report["spent_total"] == pytest.approx(0.10)


def test_verdicts_by_another_model_in_the_file_are_refused_before_any_call(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), allowed=None, first=2)
    other = _Judge()
    other.model = "another"
    with pytest.raises(ValueError, match="another model"):
        paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", other, allowed=None)
    assert other.asked == 0


def test_answers_that_changed_under_verdicts_are_refused_before_any_call(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), allowed=None, first=2)
    answers = tmp_path / "answers-oss" / "lme" / "mem0-8000" / "answers.jsonl"
    answers.write_text(answers.read_text().replace('"x"', '"y"'))
    judge = _Judge()
    with pytest.raises(ValueError, match="settings"):
        paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", judge, allowed=None)
    assert judge.asked == 0


def test_a_second_copy_of_the_job_cannot_run_beside_the_first(tmp_path: Path) -> None:
    reference = _copy(tmp_path)
    with paid_judge.only_one(tmp_path), pytest.raises(RuntimeError, match="already running"):
        paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), allowed=None)
    assert paid_judge.run_row(tmp_path, reference, "oss", "mem0-8000", _Judge(), allowed=None, first=1)["judged"] == 1


def test_the_whole_job_runs_from_the_command_line_on_a_stand_in_with_no_key(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    reference = _copy(tmp_path)
    assert paid_judge.main(["run", "--root", str(tmp_path), "--reference", str(reference), "--row", "oss/mem0-8000", "--first", "2", "--stand-in"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["judged"] == 2
    assert paid_judge.main(["status", "--root", str(tmp_path), "--stand-in"]) == 0
    assert "mem0-8000" in capsys.readouterr().out
    assert paid_judge.main(["status", "--root", str(tmp_path)]) == 0
    assert "mem0-8000" not in capsys.readouterr().out  # a stand-in's verdicts are not the paid judge's
    # The real judge is never reached with no key and no yes: it needs the key, which this refuses to run without.
    with pytest.raises(RuntimeError, match="AWS_BEARER_TOKEN_BEDROCK"):
        paid_judge.main(["run", "--root", str(tmp_path), "--reference", str(reference), "--row", "oss/mem0-8000", "--first", "1", "--part-split", str(_split(tmp_path))])


def _split(root: Path) -> Path:
    path = root / "split.json"
    path.write_text(json.dumps({"dev": [], "held_out": [f"q{n}" for n in range(10)]}))
    return path
