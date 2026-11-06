"""The last stage of the comparison: prove that a run is whole and one run, then write its tables.

``check`` reads every file a part's run wrote, and refuses the tables unless every row the protocol names is there, and only those, each
made by the one commit and copy of the code, on the part, budget, system, tokenizer and files it says, with every question answered and
judged, each answer to the prompt on record and each verdict to the answer on record, and Mem0 and the headers on the settings the protocol
names. A job only says what it was told, so this looks at what was written.

The long ingest may have come from an earlier commit than the stages after it, if what it depends on has not changed: ``--ingest-code``
names that commit's copy of the code, and it is held to ``--ingest-commit``, ``--ingest-digest`` and to this copy's ingest digest.

Rows are named ``<system>-<budget>``. Every row at 8,000 tokens goes to both readers; a row at a reader's window goes to that
reader; the budget sweep goes to gpt-oss-120b, which the protocol states.

    python -m studies.memory_bench.analyze --run runs/held_out-abc --part held_out --commit <sha> --digest <sha256> --out report.md
"""

from __future__ import annotations

import argparse
import hashlib
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
SWEEP = (2000, 4000, MAIN_BUDGET, 16000)
OSS_WINDOW = 118000  # 131,072 less the 12,288 a full-context answer may use, with room for the prompt's own frame
REFERENCE = f"taste_git-{MAIN_BUDGET}"
NOISE_PAIR = (f"flat_bm25-{MAIN_BUDGET}", f"taste_git-{MAIN_BUDGET}")
NAMED_BY_V4 = ("d6233ab6", "4dfccbf8", "ceb54acb")  # MemPalace's own BENCHMARKS.md names these as what hybrid v4 was tuned on
V4_ROW = f"mempalace_bench_v4-{MAIN_BUDGET}"
READER_SETTINGS = {("oss", "lme"): (4096, "low"), ("oss", "locomo"): (4096, "low"),
                   ("llama", "lme"): (800, None), ("llama", "locomo"): (50, None)}
FULL_CONTEXT_CAP = 12288  # on prompts over 100,000 tokens gpt-oss-120b's reasoning used up 4,096 on 8 of 50 practice questions and 8,192 on 1, leaving no answer
OSS_MODEL = "openai/gpt-oss-120b"
MEM0_SETTINGS = {"top_k": 1000, "date_prompt": True, "max_tokens": 4096, "effort": "low", "embedder": "sentence-transformers/all-MiniLM-L6-v2",
                 "mem0ai": "2.2.1", "model": OSS_MODEL}
HEADER_SETTINGS = {"model": OSS_MODEL, "max_tokens": 800, "effort": "low", "temperature": 0}
JUDGE_SETTINGS = {"model": OSS_MODEL, "max_tokens": 512, "effort": "low", "temperature": 0}
LEAST_PARSED = 0.95
MOST_EMPTY = 0.02  # a share of a row's answers that may come back empty or cut off, or a judge's verdicts unread, before the row stops counting
ARMS = {"lme": ("flat_dense", "mem0", "mempalace_bench_raw", "mempalace_bench_v4", "mempalace_product"),
        "locomo": ("flat_dense", "mem0", "mempalace_locomo_raw_session", "mempalace_locomo_raw_dialog", "mempalace_locomo_hybrid_session", "mempalace_product")}
OURS = ("flat_bm25", "taste_git", "taste_git_headers")
ALONE = {"lme": ("none-8000", "newest-8000", "full_llama-31000", f"full_oss-{OSS_WINDOW}", "oracle_llama-31000", f"oracle_oss-{OSS_WINDOW}"),
         "locomo": ("none-8000", "newest-8000", "full_llama-31000", f"full_oss-{OSS_WINDOW}")}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_rows(view: str) -> set[str]:
    """Every row the protocol names for this benchmark, and no other."""
    return {f"{arm}-{budget}" for arm in (*OURS, *ARMS[view]) for budget in SWEEP} | set(ALONE[view])


def made_with(view: str, row: str) -> dict[str, Any]:
    """What a row's records must have been made with, from its name."""
    name = row.rsplit("-", 1)[0]
    if name == "none":
        system = "none"
    elif name == "newest" or name.startswith("full_"):
        system = "full"
    elif name.startswith("oracle_"):
        system = "oracle"
    elif name in OURS:
        system = name
    else:
        system = "precomputed"
    return {"benchmark": view, "system": system, "charge": "prompt" if name.startswith(("full_", "oracle_")) else "text",
            "tokenizer": "gpt-oss-120b" if name.endswith("_oss") else "llama-3.1-70b-instruct"}


def reader_cap(reader: str, view: str, row: str) -> tuple[int, str | None]:
    """The output cap and reasoning effort a row's answers must have been asked with."""
    cap, effort = READER_SETTINGS[reader, view]
    return (FULL_CONTEXT_CAP if reader == "oss" and row.startswith("full_oss-") else cap), effort


def readers_of(row: str) -> tuple[str, ...]:
    """Which readers a row goes to, from its name."""
    name, budget = row.rsplit("-", 1)
    if name.endswith("_llama"):
        return ("llama",)
    if name.endswith("_oss"):
        return ("oss",)
    return ("llama", "oss") if int(budget) == MAIN_BUDGET else ("oss",)


def check_run(root: Path, *, part: str, commit: str, digest: str, ingest_commit: str | None = None, ingest_digest: str | None = None) -> list[str]:
    """Everything wrong with this run, in words. An empty list means the tables may be written."""
    problems: list[str] = []
    stamp = (ingest_commit or commit, ingest_digest or digest)
    for view in ("lme", "locomo"):
        folder = root / "records" / view
        if not folder.is_dir():
            problems.append(f"{view}: no records")
            continue
        present = {p.name for p in folder.iterdir() if p.is_dir()}
        problems += [f"{view}: no {row} row" for row in sorted(required_rows(view) - present)]
        problems += [f"{view}: {row} is not a row of the protocol" for row in sorted(present - required_rows(view))]
        for row in sorted(present & required_rows(view)):
            problems += _check_row(root, view, folder / row, part=part, commit=commit, digest=digest)
        problems += _check_rankings(root / "rankings" / view, view, part=part, stamp=stamp)
        problems += _check_headers(root / "headers" / f"{view}.jsonl", view, stamp=stamp)
    return problems


def _check_row(root: Path, view: str, folder: Path, *, part: str, commit: str, digest: str) -> list[str]:
    row = folder.name
    where = f"{view} {row}"
    problems = []
    try:
        settings = json.loads((folder / "records.settings.json").read_text())
        records = _rows(folder / "records.jsonl")
        json.loads((folder / "records.summary.json").read_text())  # written only when the row's own checks passed
    except FileNotFoundError as error:
        return [f"{where}: {Path(error.filename).name} is missing"]
    wanted_with = made_with(view, row)
    for key, wanted in (("commit", commit), ("code_digest", digest), ("part", part), ("budget", int(row.rsplit("-", 1)[1])),
                        ("benchmark", wanted_with["benchmark"]), ("system", wanted_with["system"]), ("charge", wanted_with["charge"])):
        if settings.get(key) != wanted:
            problems.append(f"{where}: {key} is {settings.get(key)!r}, not {wanted!r}")
    if Path(str(settings.get("tokenizer"))).name != wanted_with["tokenizer"]:
        problems.append(f"{where}: tokenizer is {settings.get('tokenizer')!r}, not {wanted_with['tokenizer']}")
    problems += _check_row_files(root, view, row, settings, where)
    if len(records) != EXPECTED[view, part]:
        problems.append(f"{where}: {len(records)} records, not {EXPECTED[view, part]}")
    if len({r["qid"] for r in records}) != len(records):
        problems.append(f"{where}: a question appears twice")
    if sum(r.get("headers_missing", 0) for r in records):
        problems.append(f"{where}: headers were lost")
    for reader in readers_of(row):
        problems += _check_answers(root, view, row, reader, records, digest=digest, commit=commit)
    if view == "lme" and row.endswith(f"-{MAIN_BUDGET}") and row != f"none-{MAIN_BUDGET}":
        problems += _check_enough(root, row, records, digest=digest, commit=commit)
    return problems


def _check_row_files(root: Path, view: str, row: str, settings: dict, where: str) -> list[str]:
    """The ranked lists and headers a row was built from must be the ones the run holds now."""
    name = row.rsplit("-", 1)[0]
    problems = []
    if made_with(view, row)["system"] == "precomputed":
        path = root / "rankings" / view / name / "rankings.jsonl"
        if not path.exists() or settings.get("rankings_sha256") != _sha256(path):
            problems.append(f"{where}: its rankings are not the ones the run holds")
    if name == "taste_git_headers":
        path = root / "headers" / f"{view}.jsonl"
        if not path.exists() or settings.get("headers_sha256") != _sha256(path):
            problems.append(f"{where}: its headers are not the ones the run holds")
    return problems


def _check_judge_settings(path: Path, where: str, *, commit: str, digest: str, **more: Any) -> list[str]:
    if not path.exists():
        return [f"{where}: {path.name} is missing"]
    settings = json.loads(path.read_text())
    from studies.memory_bench.core import DATA_SHA256

    wanted = {**JUDGE_SETTINGS, "commit": commit, "code_digest": digest, "reference_sha256": DATA_SHA256["lme"], **more}
    return [f"{where}: judge's {key} is {settings.get(key)!r}, not {value!r}" for key, value in wanted.items() if settings.get(key) != value]


def _check_enough(root: Path, row: str, records: list[dict], *, digest: str, commit: str) -> list[str]:
    where = f"lme {row} enough"
    path = root / "enough" / "lme" / row / "verdicts.jsonl"
    if not path.exists():
        return [f"{where}: no verdicts"]
    verdicts_here = {r["question_id"]: r for r in _rows(path)}
    hashes = {r["question_id"]: r["prompt_sha256"] for r in records}
    problems = []
    if set(verdicts_here) != set(hashes):
        problems.append(f"{where}: {len(set(hashes) - set(verdicts_here))} questions have no verdict")
    stale = sum(q in hashes and r.get("prompt_sha256") != hashes[q] for q, r in verdicts_here.items())
    if stale:
        problems.append(f"{where}: {stale} verdicts are of what is no longer delivered")
    unread = sum(r.get("enough") is None for r in verdicts_here.values())
    if unread > MOST_EMPTY * max(1, len(hashes)):
        problems.append(f"{where}: {unread} verdicts could not be read")
    return problems + _check_judge_settings(path.with_suffix(".settings.json"), where, commit=commit, digest=digest, max_tokens=2048)


def _check_headers(path: Path, view: str, *, stamp: tuple[str, str]) -> list[str]:
    where = f"{view} headers"
    if not path.exists():
        return [f"{where}: no file"]
    rows = _rows(path)
    if not rows:
        return [f"{where}: empty"]
    wanted = {**HEADER_SETTINGS, "study_commit": stamp[0], "code_digest": stamp[1]}
    problems = []
    for key, value in wanted.items():
        seen = {row.get(key) for row in rows}
        if seen != {value}:
            problems.append(f"{where}: {key} is {sorted(map(str, seen))}, not {value!r}")
    if len({row.get("prompt_sha256") for row in rows}) != 1 or rows[0].get("prompt_sha256") is None:
        problems.append(f"{where}: not one prompt")
    written = sum(bool(row.get("ok")) for row in rows)
    if written / len(rows) < LEAST_PARSED:
        problems.append(f"{where}: {written} of {len(rows)} parsed")
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
    cap, effort = reader_cap(reader, view, row)
    for key, wanted in (("max_tokens", cap), ("effort", effort), ("study_commit", commit), ("code_digest", digest)):
        seen = {a.get(key) for a in answers}
        if seen != {wanted}:
            problems.append(f"{where}: {key} is {sorted(map(str, seen))}, not {wanted!r}")
    empty = sum(not str(a.get("hypothesis", "")).strip() for a in answers)
    if empty > MOST_EMPTY * max(1, len(hashes)):
        problems.append(f"{where}: {empty} answers are empty")
    # Llama's caps are the benchmarks' own, and short: being stopped by one is how those answers end. A model that reasons first has nothing to show if it is.
    cut = sum(a.get("finish_reason") == "length" for a in answers)
    if reader == "oss" and cut > MOST_EMPTY * max(1, len(hashes)):
        problems.append(f"{where}: {cut} answers were cut off")
    if view == "lme":
        verdicts_path = path.with_name(path.name + ".eval-results-served")
        judged = {r["question_id"]: r for r in _rows(verdicts_path)} if verdicts_path.exists() else {}
        if set(judged) != set(hashes):
            problems.append(f"{where}: {len(set(hashes) - set(judged))} answers have no verdict")
        stale = sum(q in said and (r.get("hypothesis") != said[q].get("hypothesis") or r.get("prompt_sha256") != said[q].get("prompt_sha256"))
                    for q, r in judged.items())
        if stale:
            problems.append(f"{where}: {stale} verdicts are of another answer")
        unread = sum(r["autoeval_label"].get("label") is None for r in judged.values())
        if unread > MOST_EMPTY * max(1, len(hashes)):
            problems.append(f"{where}: {unread} verdicts could not be read")
        if judged:
            problems += _check_judge_settings(Path(f"{verdicts_path}.settings.json"), where, commit=commit, digest=digest, answers_sha256=_sha256(path))
    else:
        scores = path.with_name("scores.jsonl")
        if not scores.exists() or len(_rows(scores)) != len(hashes):
            problems.append(f"{where}: not scored")
        else:
            summary_path = path.with_name("scores.summary.json")
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if not summary.get("audit_sha256"):
                problems.append(f"{where}: scored with no audit file")
            if summary.get("answers_sha256") != _sha256(path):
                problems.append(f"{where}: scored from other answers")
            for key, wanted in (("commit", commit), ("code_digest", digest)):
                if summary.get(key) != wanted:
                    problems.append(f"{where}: scored by {key} {summary.get(key)!r}, not {wanted!r}")
    return problems


def _check_rankings(folder: Path, view: str, *, part: str, stamp: tuple[str, str]) -> list[str]:
    problems: list[str] = []
    if not folder.is_dir():
        return [f"{view}: no rankings"]
    present = {p.name for p in folder.iterdir() if p.is_dir()}
    problems += [f"{view} {arm}: no rankings" for arm in sorted(set(ARMS[view]) - present)]
    problems += [f"{view} {arm}: rankings of no row in the protocol" for arm in sorted(present - set(ARMS[view]))]
    for arm in sorted(folder / name for name in present & set(ARMS[view])):
        path = arm / "rankings.jsonl"
        if not path.exists():
            problems.append(f"{view} {arm.name}: no rankings file")
            continue
        rows = _rows(path)
        versions = [row.get("versions", {}) for row in rows]
        if len(rows) != EXPECTED[view, part]:
            problems.append(f"{view} {arm.name}: {len(rows)} rankings, not {EXPECTED[view, part]}")
        if len({row.get("qid") for row in rows}) != len(rows):
            problems.append(f"{view} {arm.name}: a question was ranked twice")
        for key, wanted in (("study_commit", stamp[0]), ("code_digest", stamp[1])):
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


def kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa of two judges over the same answers."""
    n = len(a)
    if not n or n != len(b):
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    first, second = sum(a) / n, sum(b) / n
    chance = first * second + (1 - first) * (1 - second)
    return (observed - chance) / (1 - chance) if chance < 1 else None


def judges_agree(root: Path, reader: str, other: str) -> dict[str, Any] | None:
    """The free judge against a second one, over every answer both have read: agreement and kappa."""
    folder = root / f"answers-{reader}" / "lme"
    first: list[bool] = []
    second: list[bool] = []
    for row in sorted(p for p in folder.iterdir() if p.is_dir()):
        theirs = row / f"answers.jsonl.eval-results-{other}"
        if theirs.exists():
            a, b = verdicts(row / "answers.jsonl.eval-results-served"), verdicts(theirs)
            for question in sorted(set(a) & set(b)):
                first.append(a[question])
                second.append(b[question])
    if not first:
        return None
    return {"judge": other, "answers": len(first), "agree": sum(x == y for x, y in zip(first, second, strict=True)), "kappa": kappa(first, second)}


def cost_ledger(root: Path) -> dict[str, Any]:
    """Every stage's model tokens by the server's own counters, ingest and reading apart, and GPU-hours from the minute-by-minute record each job keeps. A stage that shared a
    server with other work counts that work too, so a rehearsal's figures are upper bounds."""
    stages: dict[str, dict[str, float]] = {}
    for path in sorted(root.rglob("vllm-metrics.txt")):
        top = path.relative_to(root).parts[0]
        stage = stages.setdefault(top, {"prompt_tokens": 0, "generated_tokens": 0, "cached_tokens": 0, "calls": 0, "gpu_hours": 0.0})
        for line in path.read_text().splitlines():
            name, _, value = line.rpartition(" ")
            if name.startswith("vllm:prompt_tokens_total"):
                stage["prompt_tokens"] += float(value)
            elif name.startswith("vllm:generation_tokens_total"):
                stage["generated_tokens"] += float(value)
            elif name.startswith("vllm:prefix_cache_hits_total"):
                stage["cached_tokens"] += float(value)
            elif name.startswith("vllm:request_success_total"):
                stage["calls"] += float(value)
        gpu = path.with_name("gpu.csv")
        if gpu.exists():
            samples = gpu.read_text().splitlines()[1:]
            stage["gpu_hours"] += len(samples) / 60
    return stages


def _reference_row(row: str, rows: Iterable[str]) -> str:
    """Each row is read against Taste with headers off at its own budget; a row with no budget of its own, against the main one."""
    own = f"taste_git-{row.rsplit('-', 1)[1]}"
    return own if own in rows else REFERENCE


def _empty_and_cut(path: Path) -> dict[str, int]:
    answers = _rows(path)
    return {"empty": sum(not str(a.get("hypothesis", "")).strip() for a in answers), "cut_off": sum(a.get("finish_reason") == "length" for a in answers)}


def _base(question_id: str) -> str:
    return question_id.removesuffix("_abs")


def lme_table(root: Path, reader: str, judge: str = "served") -> list[dict[str, Any]]:
    folder = root / f"answers-{reader}" / "lme"
    records = {row.name: {r["question_id"]: r for r in _rows(root / "records" / "lme" / row.name / "records.jsonl")}
               for row in sorted(folder.iterdir()) if row.is_dir()}
    said = {row: verdicts(folder / row / f"answers.jsonl.eval-results-{judge}") for row in records}
    table = []
    for row, right in said.items():
        rows = records[row]
        entry: dict[str, Any] = {"row": row, "n": len(right), "yes": sum(right.values()), "accuracy": statistics.fmean(right.values()),
                                 "tokens": statistics.fmean(r["tokens_delivered"] for r in rows.values()),
                                 "dropped": statistics.fmean(r["dropped"] for r in rows.values()), **_empty_and_cut(folder / row / "answers.jsonl")}
        official = [r["autoeval_label"].get("official_rule_label") for r in _rows(folder / row / f"answers.jsonl.eval-results-{judge}")]
        entry["accuracy_official_rule"] = statistics.fmean(value is True for value in official) if official and None not in official else None
        kinds: dict[str, list[bool]] = {}
        for question, value in right.items():
            kinds.setdefault(rows[question]["question_type"], []).append(value)
        entry["by_type"] = {kind: statistics.fmean(values) for kind, values in sorted(kinds.items())}
        scored = [r["scores"] for r in rows.values() if r["scores"]]
        for key in (*(f"session_recall_{kind}@{cut}" for cut in (1, 3, 5, 10, 30, 50) for kind in ("any", "all")), "session_ndcg@10",
                    "session_recall_any@budget", "session_recall_all@budget", "round_recall_any@budget", "round_recall_all@budget", "evidence_survived"):
            values = [s[key] for s in scored if key in s]
            entry[key] = statistics.fmean(values) if values else None
            entry[f"n {key}"] = len(values)
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
        base = said.get(_reference_row(row, said))
        if base is not None and row != _reference_row(row, said):
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
            without = [rows[q]["scores"] for q in kept if rows[q]["scores"]]
            entry["without_named_questions"] = {"n": len(kept), "named_here": len(right) - len(kept), "accuracy": statistics.fmean(kept.values()),
                                                **{key: statistics.fmean(s[key] for s in without if key in s) if any(key in s for s in without) else None
                                                   for key in ("session_recall_any@5", "session_recall_any@10", "session_ndcg@10")}}
        table.append(entry)
    return table


def locomo_table(root: Path, reader: str) -> list[dict[str, Any]]:
    folder = root / f"answers-{reader}" / "locomo"
    scores = {row.name: {r["question_id"]: r for r in _rows(row / "scores.jsonl")} for row in sorted(folder.iterdir()) if row.is_dir()}
    table = []
    for row, rows in scores.items():
        summary = json.loads((folder / row / "scores.summary.json").read_text())
        records = _rows(root / "records" / "locomo" / row / "records.jsonl")
        entry: dict[str, Any] = {"row": row, "n": summary["scored_1_to_4"], "f1": summary["f1_categories_1_to_4"],
                                 "f1_without_audit_flags": summary.get("f1_without_audit_flags"),
                                 "per_category": summary["per_category"], "adversarial": summary["adversarial"],
                                 "tokens": statistics.fmean(r["tokens_delivered"] for r in records), **_empty_and_cut(folder / row / "answers.jsonl")}
        kinds: dict[str, dict[str, list[float]]] = {}
        for record in records:
            for key in (*(f"session_recall@{cut}" for cut in (1, 3, 5, 10, 30, 50)), "session_recall@budget", "turn_recall@budget"):
                if key in record["scores"]:
                    kinds.setdefault(key, {}).setdefault(str(record["category"]), []).append(record["scores"][key])
        entry["retrieval_by_category"] = {key: {c: statistics.fmean(v) for c, v in sorted(by.items())} for key, by in kinds.items()}
        entry["retrieval"] = {key: statistics.fmean(v for values in by.values() for v in values) for key, by in kinds.items()}
        base = scores.get(_reference_row(row, scores))
        if base is not None and row != _reference_row(row, scores):
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
        out += _lines(("row", "n", "yes", "accuracy", "official rule", "tokens", "empty", "cut off", "session any@1", "any@3", "any@5", "any@10", "any@30", "any@50", "session all@budget", "round all@budget (n)", "evidence kept", "enough",
                       "right/right", "right/wrong", "wrong/right", "wrong/wrong", "against Taste", "p"),
                      [(e["row"], e["n"], e["yes"], e["accuracy"], e["accuracy_official_rule"], round(e["tokens"]), e["empty"], e["cut_off"],
                        e["session_recall_any@1"], e["session_recall_any@3"], e["session_recall_any@5"], e["session_recall_any@10"], e["session_recall_any@30"], e["session_recall_any@50"],
                        e["session_recall_all@budget"], f"{_number(e['round_recall_all@budget'])} ({e['n round_recall_all@budget']})", e["evidence_survived"], e.get("enough_to_answer"), *e["retrieval_by_answer"].values(),
                        _number(e.get("against_reference", {}).get("difference")), _number(e.get("against_reference", {}).get("p")))
                       for e in lme])
        v4 = [e["without_named_questions"] for e in lme if "without_named_questions" in e]
        out += ["", f"v4 without the questions it was tuned on: {v4}", ""]
        data.setdefault("judges", {})[reader] = judges_agree(root, reader, "gpt-5.4")
        out += [f"The second judge, reader {reader}: {data['judges'][reader] or 'not run'}", "", f"## LoCoMo, reader {reader}, judge-free F1", ""]
        out += _lines(("row", "n", "F1", "F1 without audit flags", "tokens", "empty", "cut off", "temporal", "adversarial", "session@1", "@3", "@5", "@10", "@30", "@50", "session@budget", "turn@budget", "against Taste"),
                      [(e["row"], e["n"], e["f1"], e["f1_without_audit_flags"], round(e["tokens"]), e["empty"], e["cut_off"], e["per_category"].get("2"), e["adversarial"],
                        *(e["retrieval"].get(f"session_recall@{cut}") for cut in (1, 3, 5, 10, 30, 50)),
                        e["retrieval"].get("session_recall@budget"), e["retrieval"].get("turn_recall@budget"),
                        _number(e.get("against_reference", {}).get("difference"))) for e in locomo])
        out += ["", f"Same prompts sent twice, reader {reader}: {data['noise'][reader]}", ""]
    data["ledger"] = cost_ledger(root)
    out += ["## The ledger, by the server's own counters", "", f"{data['ledger']}", ""]
    return "\n".join(out) + "\n", data


def _check_ingest_code(commit: str | None, digest: str | None, code: str | None) -> list[str]:
    """The earlier commit's copy of the code is that commit's, and what the long ingest depends on is the same in this copy."""
    from studies.memory_ledger import code_digest, ingest_digest

    if not (commit and digest and code):
        return ["--ingest-commit, --ingest-digest and --ingest-code go together"]
    problems = []
    if code_digest(Path(code)) != digest:
        problems.append(f"{code} is not the code of {commit}")
    if ingest_digest(Path(code)) != ingest_digest():
        problems.append(f"what the long ingest depends on has changed since {commit}: the ingest must be run again")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("run", "commit", "digest", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--part", required=True, choices=("dev", "held_out"))
    for name in ("ingest-commit", "ingest-digest", "ingest-code"):
        parser.add_argument(f"--{name}", help="an earlier commit's long ingest: all three, or none")
    arguments = parser.parse_args(argv)
    root = Path(arguments.run)
    earlier = (arguments.ingest_commit, arguments.ingest_digest, arguments.ingest_code)
    problems = _check_ingest_code(*earlier) if any(earlier) else []
    problems += check_run(root, part=arguments.part, commit=arguments.commit, digest=arguments.digest, ingest_commit=arguments.ingest_commit, ingest_digest=arguments.ingest_digest)
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
