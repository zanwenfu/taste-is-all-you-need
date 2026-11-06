"""The last stage of the comparison: prove that a run is whole and one run, then write its tables.

``check`` reads every file a part's run wrote, and refuses the tables unless each row was made by the one commit and copy of the
code, on the part and budget it says, with every question answered and judged, each answer to the prompt that is on record, and
Mem0 and the headers on the settings the protocol names. A job only says what it was told, so this looks at what was written.

Rows are named ``<system>-<budget>``. Every row at 8,000 tokens goes to both readers; a row at a reader's window goes to that
reader; the budget sweep goes to gpt-oss-120b, which the protocol states.

    python -m studies.memory_bench.analyze --run runs/held_out-abc --part held_out --commit <sha> --digest <sha256> --out report.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

EXPECTED = {("lme", "dev"): 50, ("lme", "held_out"): 450, ("locomo", "dev"): 435, ("locomo", "held_out"): 1551}
MAIN_BUDGET = 8000
REFERENCE = f"taste_git-{MAIN_BUDGET}"
NOISE_PAIR = (f"flat_bm25-{MAIN_BUDGET}", f"taste_git-{MAIN_BUDGET}")
NAMED_BY_V4 = ("d6233ab6", "4dfccbf8", "ceb54acb")  # MemPalace's own BENCHMARKS.md names these as what hybrid v4 was tuned on
V4_ROW = f"mempalace_bench_v4-{MAIN_BUDGET}"
READER_SETTINGS = {("oss", "lme"): (4096, "low"), ("oss", "locomo"): (2048, "low"),
                   ("llama", "lme"): (800, None), ("llama", "locomo"): (50, None)}
MEM0_SETTINGS = {"top_k": 1000, "date_prompt": True, "max_tokens": 4096, "effort": "low"}
LEAST_PARSED = 0.95


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def readers_of(row: str) -> tuple[str, ...]:
    """Which readers a row goes to, from its name."""
    name, budget = row.rsplit("-", 1)
    if name.endswith("_llama"):
        return ("llama",)
    if name.endswith("_oss"):
        return ("oss",)
    return ("llama", "oss") if int(budget) == MAIN_BUDGET else ("oss",)


def check_run(root: Path, *, part: str, commit: str, digest: str) -> list[str]:
    """Everything wrong with this run, in words. An empty list means the tables may be written."""
    problems: list[str] = []
    for view in ("lme", "locomo"):
        folder = root / "records" / view
        if not folder.is_dir():
            problems.append(f"{view}: no records")
            continue
        for row_folder in sorted(p for p in folder.iterdir() if p.is_dir()):
            problems += _check_row(root, view, row_folder, part=part, commit=commit, digest=digest)
        problems += _check_rankings(root / "rankings" / view, view, commit=commit, digest=digest)
        if not (folder / REFERENCE).is_dir():
            problems.append(f"{view}: no {REFERENCE} row")
    return problems


def _check_row(root: Path, view: str, folder: Path, *, part: str, commit: str, digest: str) -> list[str]:
    row = folder.name
    where = f"{view} {row}"
    problems = []
    try:
        settings = json.loads((folder / "records.settings.json").read_text())
        records = _rows(folder / "records.jsonl")
    except FileNotFoundError as error:
        return [f"{where}: {Path(error.filename).name} is missing"]
    for key, wanted in (("commit", commit), ("code_digest", digest), ("part", part), ("budget", int(row.rsplit("-", 1)[1]))):
        if settings.get(key) != wanted:
            problems.append(f"{where}: {key} is {settings.get(key)!r}, not {wanted!r}")
    if len(records) != EXPECTED[view, part]:
        problems.append(f"{where}: {len(records)} records, not {EXPECTED[view, part]}")
    if len({r["qid"] for r in records}) != len(records):
        problems.append(f"{where}: a question appears twice")
    if sum(r.get("headers_missing", 0) for r in records):
        problems.append(f"{where}: headers were lost")
    for reader in readers_of(row):
        problems += _check_answers(root, view, row, reader, records, digest=digest, commit=commit)
    return problems


def _check_answers(root: Path, view: str, row: str, reader: str, records: list[dict], *, digest: str, commit: str) -> list[str]:
    where = f"{view} {row} {reader}"
    path = root / f"answers-{reader}" / view / row / "answers.jsonl"
    if not path.exists():
        return [f"{where}: no answers"]
    answers = _rows(path)
    problems = []
    hashes = {r["question_id"]: r["prompt_sha256"] for r in records}
    said = {a["question_id"]: a for a in answers}
    if len(said) != len(answers):
        problems.append(f"{where}: a question was answered twice")
    if set(said) != set(hashes):
        problems.append(f"{where}: {len(set(hashes) - set(said))} questions have no answer, {len(set(said) - set(hashes))} are not asked")
    stale = [q for q, a in said.items() if q in hashes and a.get("prompt_sha256") != hashes[q]]
    if stale:
        problems.append(f"{where}: {len(stale)} answers are to prompts no longer on record")
    cap, effort = READER_SETTINGS[reader, view]
    for key, wanted in (("max_tokens", cap), ("effort", effort), ("study_commit", commit), ("code_digest", digest)):
        seen = {a.get(key) for a in answers}
        if seen != {wanted}:
            problems.append(f"{where}: {key} is {sorted(map(str, seen))}, not {wanted!r}")
    if view == "lme":
        verdicts = path.with_name(path.name + ".eval-results-served")
        judged = {r["question_id"] for r in _rows(verdicts)} if verdicts.exists() else set()
        if judged != set(hashes):
            problems.append(f"{where}: {len(set(hashes) - judged)} answers have no verdict")
    else:
        scores = path.with_name("scores.jsonl")
        if not scores.exists() or len(_rows(scores)) != len(hashes):
            problems.append(f"{where}: not scored")
        else:
            summary = path.with_name("scores.summary.json")
            if not summary.exists() or not json.loads(summary.read_text()).get("audit_sha256"):
                problems.append(f"{where}: scored with no audit file")
    return problems


def _check_rankings(folder: Path, view: str, *, commit: str, digest: str) -> list[str]:
    problems: list[str] = []
    if not folder.is_dir():
        return [f"{view}: no rankings"]
    for arm in sorted(p for p in folder.iterdir() if p.is_dir()):
        path = arm / "rankings.jsonl"
        if not path.exists():
            problems.append(f"{view} {arm.name}: no rankings file")
            continue
        versions = [row.get("versions", {}) for row in _rows(path)]
        for key, wanted in (("study_commit", commit), ("code_digest", digest)):
            seen = {v.get(key) for v in versions}
            if seen != {wanted}:
                problems.append(f"{view} {arm.name}: {key} is {sorted(map(str, seen))}, not {wanted!r}")
        if arm.name == "mem0":
            for key, wanted in MEM0_SETTINGS.items():
                seen = {v.get(key) for v in versions}
                if seen != {wanted}:
                    problems.append(f"{view} mem0: {key} is {sorted(map(str, seen))}, not {wanted!r}")
            replies = [r for p in sorted(arm.glob("shard*.replies.jsonl")) for r in _rows(p)]
            parsed = sum(bool(r["parsed"]) for r in replies)
            if not replies or parsed / len(replies) < LEAST_PARSED:
                problems.append(f"{view} mem0: {parsed} of {len(replies)} replies parsed")
    return problems


# ------------------------------------------------------------------ the numbers


def sign_test(better: int, worse: int) -> float:
    n = better + worse
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / 2**n) if n else 1.0


def bootstrap_difference(pairs: dict[str, list[tuple[float, float]]], *, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean of b - a, and its 95% interval), drawing whole clusters with replacement."""
    names = sorted(pairs)
    flat = [pair for name in names for pair in pairs[name]]
    mean = statistics.fmean(b - a for a, b in flat)
    generator = random.Random(seed)
    means = []
    for _ in range(draws):
        picked = [pair for _ in names for pair in pairs[generator.choice(names)]]
        means.append(statistics.fmean(b - a for a, b in picked))
    means.sort()
    return mean, means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def verdicts(path: Path) -> dict[str, bool]:
    """Yes or no per question; a verdict that cannot be read counts as no, and is counted."""
    return {row["question_id"]: row["autoeval_label"]["label"] is True for row in _rows(path)}


def _base(question_id: str) -> str:
    return question_id.removesuffix("_abs")


def lme_table(root: Path, reader: str, judge: str = "served") -> list[dict[str, Any]]:
    folder = root / f"answers-{reader}" / "lme"
    records = {row.name: {r["question_id"]: r for r in _rows(root / "records" / "lme" / row.name / "records.jsonl")}
               for row in sorted(folder.iterdir()) if row.is_dir()}
    said = {row: verdicts(folder / row / f"answers.jsonl.eval-results-{judge}") for row in records}
    base = said.get(REFERENCE)
    table = []
    for row, right in said.items():
        rows = records[row]
        entry: dict[str, Any] = {"row": row, "n": len(right), "yes": sum(right.values()), "accuracy": statistics.fmean(right.values()),
                                 "tokens": statistics.fmean(r["tokens_delivered"] for r in rows.values()),
                                 "dropped": statistics.fmean(r["dropped"] for r in rows.values())}
        kinds: dict[str, list[bool]] = {}
        for question, value in right.items():
            kinds.setdefault(rows[question]["question_type"], []).append(value)
        entry["by_type"] = {kind: statistics.fmean(values) for kind, values in sorted(kinds.items())}
        scored = [r["scores"] for r in rows.values() if r["scores"]]
        for key in ("session_recall_any@5", "session_recall_any@10", "session_recall_all@10", "session_ndcg@10",
                    "session_recall_all@budget", "round_recall_all@budget", "evidence_survived"):
            values = [s[key] for s in scored if key in s]
            entry[key] = statistics.fmean(values) if values else None
        grid = {"right_right": 0, "right_wrong": 0, "wrong_right": 0, "wrong_wrong": 0}
        for question, value in right.items():
            found = rows[question]["scores"].get("session_recall_all@budget")
            if found is not None:
                grid[f"{'right' if found else 'wrong'}_{'right' if value else 'wrong'}"] += 1
        entry["retrieval_by_answer"] = grid
        judged = root / "enough" / "lme" / row / "verdicts.jsonl"
        if judged.exists():
            enough = {r["question_id"]: r["enough"] for r in _rows(judged)}
            entry["enough_to_answer"] = sum(v is True for v in enough.values()) / len(enough)
            grid = {"enough_right": 0, "enough_wrong": 0, "short_right": 0, "short_wrong": 0}
            for question, value in right.items():
                grid[f"{'enough' if enough.get(question) else 'short'}_{'right' if value else 'wrong'}"] += 1
            entry["enough_by_answer"] = grid
        if base is not None and row != REFERENCE:
            shared = sorted(set(base) & set(right))
            better = sum(right[q] and not base[q] for q in shared)
            worse = sum(base[q] and not right[q] for q in shared)
            pairs: dict[str, list[tuple[float, float]]] = {}
            for question in shared:
                pairs.setdefault(_base(question), []).append((float(base[question]), float(right[question])))
            mean, low, high = bootstrap_difference(pairs)
            entry["against_reference"] = {"better": better, "worse": worse, "p": sign_test(better, worse), "difference": mean,
                                          "interval": [low, high]}
        if row == V4_ROW:
            kept = {q: value for q, value in right.items() if q not in NAMED_BY_V4}
            entry["without_named_questions"] = {"n": len(kept), "named_here": len(right) - len(kept),
                                                "accuracy": statistics.fmean(kept.values())}
        table.append(entry)
    return table


def locomo_table(root: Path, reader: str) -> list[dict[str, Any]]:
    folder = root / f"answers-{reader}" / "locomo"
    scores = {row.name: {r["question_id"]: r for r in _rows(row / "scores.jsonl")} for row in sorted(folder.iterdir()) if row.is_dir()}
    base = scores.get(REFERENCE)
    table = []
    for row, rows in scores.items():
        summary = json.loads((folder / row / "scores.summary.json").read_text())
        records = _rows(root / "records" / "locomo" / row / "records.jsonl")
        entry: dict[str, Any] = {"row": row, "n": summary["scored_1_to_4"], "f1": summary["f1_categories_1_to_4"],
                                 "f1_without_audit_flags": summary.get("f1_without_audit_flags"),
                                 "per_category": summary["per_category"], "adversarial": summary["adversarial"],
                                 "tokens": statistics.fmean(r["tokens_delivered"] for r in records)}
        kinds: dict[str, dict[str, list[float]]] = {}
        for record in records:
            for key in ("session_recall@10", "session_recall@budget", "turn_recall@budget"):
                if key in record["scores"]:
                    kinds.setdefault(key, {}).setdefault(str(record["category"]), []).append(record["scores"][key])
        entry["retrieval_by_category"] = {key: {c: statistics.fmean(v) for c, v in sorted(by.items())} for key, by in kinds.items()}
        entry["retrieval"] = {key: statistics.fmean(v for values in by.values() for v in values) for key, by in kinds.items()}
        if base is not None and row != REFERENCE:
            pairs: dict[str, list[tuple[float, float]]] = {}
            for question, record in rows.items():
                if record["category"] != 5:
                    pairs.setdefault(question.split(":")[0], []).append((base[question]["score"], record["score"]))
            mean, low, high = bootstrap_difference(pairs)
            entry["against_reference"] = {"difference": mean, "interval": [low, high], "conversations": len(pairs)}
        table.append(entry)
    return table


def noise_floor(root: Path, reader: str) -> dict[str, Any]:
    """The keyword list and Taste with headers off send the same prompts, so what differs between their answers is the reader's own
    noise. It is the floor under every gap in the tables."""
    out: dict[str, Any] = {}
    for view in ("lme", "locomo"):
        folder = root / f"answers-{reader}" / view
        first, second = (folder / row / "answers.jsonl" for row in NOISE_PAIR)
        if not (first.exists() and second.exists()):
            continue
        a, b = ({r["question_id"]: r for r in _rows(path)} for path in (first, second))
        same_prompt = sum(a[q]["prompt_sha256"] == b[q]["prompt_sha256"] for q in a)
        same_words = sum(a[q]["hypothesis"] == b[q]["hypothesis"] for q in a if a[q]["prompt_sha256"] == b[q]["prompt_sha256"])
        out[view] = {"questions": len(a), "same_prompt": same_prompt, "same_answer_given_same_prompt": same_words}
        if view == "lme":
            x, y = (verdicts(path.with_name(path.name + ".eval-results-served")) for path in (first, second))
            out[view]["verdicts_differ"] = sum(x[q] != y[q] for q in x)
    return out


# ------------------------------------------------------------------ the report


def _number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _lines(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(_number(cell) for cell in row) + " |" for row in rows)]


def report(root: Path, readers: Sequence[str] = ("llama", "oss")) -> tuple[str, dict[str, Any]]:
    data: dict[str, Any] = {"lme": {}, "locomo": {}, "noise": {}}
    out = [f"# Comparison tables: {root.name}", ""]
    for reader in readers:
        if not (root / f"answers-{reader}").is_dir():
            continue
        lme, locomo = lme_table(root, reader), locomo_table(root, reader)
        data["lme"][reader], data["locomo"][reader], data["noise"][reader] = lme, locomo, noise_floor(root, reader)
        out += [f"## LongMemEval, reader {reader}, free judge", ""]
        out += _lines(("row", "n", "yes", "accuracy", "tokens", "session any@10", "session all@budget", "round all@budget", "evidence kept", "enough",
                       "right/right", "right/wrong", "wrong/right", "wrong/wrong", "against Taste", "p"),
                      [(e["row"], e["n"], e["yes"], e["accuracy"], round(e["tokens"]), e["session_recall_any@10"], e["session_recall_all@budget"],
                        e["round_recall_all@budget"], e["evidence_survived"], e.get("enough_to_answer"), *e["retrieval_by_answer"].values(),
                        _number(e.get("against_reference", {}).get("difference")), _number(e.get("against_reference", {}).get("p")))
                       for e in lme])
        v4 = [e["without_named_questions"] for e in lme if "without_named_questions" in e]
        out += ["", f"v4 without the questions it was tuned on: {v4}", "", f"## LoCoMo, reader {reader}, judge-free F1", ""]
        out += _lines(("row", "n", "F1", "F1 without audit flags", "tokens", "temporal", "adversarial", "session@budget", "turn@budget", "against Taste"),
                      [(e["row"], e["n"], e["f1"], e["f1_without_audit_flags"], round(e["tokens"]), e["per_category"].get("2"), e["adversarial"],
                        e["retrieval"].get("session_recall@budget"), e["retrieval"].get("turn_recall@budget"),
                        _number(e.get("against_reference", {}).get("difference"))) for e in locomo])
        out += ["", f"Same prompts sent twice, reader {reader}: {data['noise'][reader]}", ""]
    return "\n".join(out) + "\n", data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("run", "commit", "digest", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--part", required=True, choices=("dev", "held_out"))
    arguments = parser.parse_args(argv)
    root = Path(arguments.run)
    problems = check_run(root, part=arguments.part, commit=arguments.commit, digest=arguments.digest)
    if problems:
        print(f"{len(problems)} problems; no tables written:")
        print("\n".join(problems))
        return 1
    text, data = report(root)
    out = Path(arguments.out)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(data, indent=1))
    print(f"checked whole; tables at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
