"""The command line of the comparison: headers, retrieve, read, judge. No stage calls a paid model.

    python -m studies.memory_bench.run retrieve --data S.json --split data/splits/memory-bench-lme.json --part dev \\
        --system taste_git --budget 8000 --tokenizer <model folder> --workdir /tmp/x --out runs/x/records.jsonl
    python -m studies.memory_bench.run read --records runs/x/records.jsonl --url http://127.0.0.1:8000/v1 --model NAME \\
        --out runs/x/answers.jsonl
    python -m studies.memory_bench.run judge --answers runs/x/answers.jsonl --reference ORACLE.json \\
        --url http://127.0.0.1:8000/v1 --model NAME

The held-out part runs once, on settings committed beforehand: ``--frozen-config`` must name the commit in use.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any

from studies.memory_bench import lme, locomo
from studies.memory_bench.headers import served_ask, stored_writer, write_all
from studies.memory_bench.reader import ServedJudge, read_all
from studies.memory_bench.runner import retrieve_conversation, retrieve_one, validate
from studies.memory_bench.systems import FlatBm25, FullContext, NoMemory, Oracle, System, TasteGit
from studies.memory_ledger import code_version


def make_system(name: str, gold: lme.Gold | None, workdir: Path, *, unit: str = "round", commit_per: str = "session",
                headers: Path | None = None) -> System:
    if name == "none":
        return NoMemory()
    if name == "full":
        return FullContext()
    if name == "oracle":
        if gold is None:
            raise ValueError("the oracle exists only for LongMemEval")
        return Oracle(gold)
    if name == "flat_bm25":
        return FlatBm25(unit=unit)
    if name == "taste_git":
        return TasteGit(workdir, unit=unit, commit_per=commit_per)
    if name == "taste_git_headers":
        if headers is None:
            raise ValueError("taste_git_headers needs stored headers")
        return TasteGit(workdir, unit=unit, commit_per=commit_per, header_writer=stored_writer(headers))
    raise ValueError(f"unknown system {name!r}")


@lru_cache(maxsize=1)
def _counter(tokenizer: str) -> Callable[[str], int]:
    """The one tokenizer that counts every budget. "words" is for tests and dry runs only."""
    if tokenizer == "words":
        return lambda text: len(text.split())
    from transformers import AutoTokenizer

    loaded = AutoTokenizer.from_pretrained(tokenizer)
    return lambda text: len(loaded(text, add_special_tokens=False)["input_ids"])


def _one(job: tuple[dict, Any, dict[str, Any]]) -> list[dict[str, Any]]:
    """One LongMemEval question, or one LoCoMo conversation with all of its questions."""
    view, gold, settings = job
    single = settings["benchmark"] == "lme"
    system = make_system(settings["system"], gold if single else None, Path(settings["workdir"]), unit=settings["unit"],
                         commit_per=settings["commit_per"],
                         headers=Path(settings["headers"]) if settings["headers"] else None)
    count = _counter(settings["tokenizer"])
    if single:
        return [retrieve_one(view, gold, system, budget=settings["budget"], count=count)]
    return retrieve_conversation(view, gold, system, budget=settings["budget"], count=count)


def _pool(workers: int) -> Executor:
    # Forking is what lets each question have its own process on Linux, where the runs happen.
    if sys.platform == "linux":
        return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"))
    return ThreadPoolExecutor(workers)


def _summary(rows: list[dict[str, Any]], settings: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row["scores"]})
    return {
        **{key: settings[key] for key in ("benchmark", "system", "part", "budget", "tokenizer", "unit", "commit_per")},
        "questions": len(rows),
        "ledger": ledger,
        "means": {key: fmean(row["scores"][key] for row in rows if key in row["scores"]) for key in keys},
        "scored_questions": sum(bool(row["scores"]) for row in rows),
        "tokens_delivered_mean": fmean(row["tokens_delivered"] for row in rows),
        "items_cut": sum(row["cut"] for row in rows),
        "items_dropped_mean": fmean(row["dropped"] for row in rows),
        "ingest_seconds_mean": fmean(row["ingest_seconds"] for row in rows),
        "retrieve_seconds_mean": fmean(row["retrieve_seconds"] for row in rows),
    }


def retrieve(arguments: argparse.Namespace) -> int:
    if arguments.part == "held_out":
        commit, dirty = code_version()
        if arguments.frozen_config != commit or dirty:
            raise SystemExit("held-out runs once, on committed settings: --frozen-config must name the clean commit in use")
    wanted = set(json.loads(Path(arguments.split).read_text())[arguments.part])
    field, loader = ("question_id", lme.load) if arguments.benchmark == "lme" else ("sample_id", locomo.load)
    views, golds = loader([item for item in json.loads(Path(arguments.data).read_text()) if item[field] in wanted])
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.workdir).mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["qid"] for line in out.read_text().splitlines()} if out.exists() else set()
    settings = {key: getattr(arguments, key) for key in
                ("benchmark", "system", "part", "budget", "tokenizer", "unit", "commit_per", "workdir", "headers")}
    if arguments.benchmark == "lme":
        jobs = [(view, golds[view["qid"]], settings) for view in views if view["qid"] not in done]
    else:
        # A conversation's records are written together, so one is either all here or to be run again.
        jobs = [(view, golds, settings) for view in views if view["questions"][0]["qid"] not in done]
    with _pool(arguments.workers) as pool, out.open("a") as written:
        for future in as_completed([pool.submit(_one, job) for job in jobs]):
            written.write("".join(json.dumps(record) + "\n" for record in future.result()))
            written.flush()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    ledger = validate(rows, expected=set(golds))
    summary = _summary(rows, settings, ledger)
    out.with_name(out.stem + ".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({key: summary[key] for key in ("system", "part", "questions", "tokens_delivered_mean")}))
    return 0


def _client(url: str) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=url, api_key="EMPTY", timeout=3600)


def headers(arguments: argparse.Namespace) -> int:
    wanted = set(json.loads(Path(arguments.split).read_text())[arguments.part])
    views, _ = lme.load([item for item in json.loads(Path(arguments.data).read_text()) if item["question_id"] in wanted])
    ask = served_ask(_client(arguments.url), arguments.model, max_tokens=arguments.max_tokens, effort=arguments.effort)
    print(json.dumps(write_all(views, ask, Path(arguments.out), workers=arguments.workers)))
    return 0


def read(arguments: argparse.Namespace) -> int:
    ask = served_ask(_client(arguments.url), arguments.model, max_tokens=arguments.max_tokens, effort=arguments.effort)
    print(json.dumps(read_all(Path(arguments.records), ask, Path(arguments.out), workers=arguments.workers)))
    return 0


def judge(arguments: argparse.Namespace) -> int:
    from studies.context_rot.lme_judge import judge_file

    served = ServedJudge(_client(arguments.url), arguments.model)
    print(judge_file(arguments.answers, arguments.reference, served, short_name="served", workers=arguments.workers))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    stage = stages.add_parser("retrieve")
    for name in ("data", "split", "system", "tokenizer", "workdir", "out"):
        stage.add_argument(f"--{name}", required=True)
    stage.add_argument("--part", required=True, choices=("dev", "held_out"))
    stage.add_argument("--benchmark", default="lme", choices=("lme", "locomo"))
    stage.add_argument("--budget", required=True, type=int)
    stage.add_argument("--unit", default="round", choices=("round", "session"))
    stage.add_argument("--commit-per", default="session", choices=("round", "session"))
    stage.add_argument("--headers")
    stage.add_argument("--frozen-config")
    stage.add_argument("--workers", type=int, default=8)
    stage.set_defaults(run=retrieve)

    for name, run, needed in (("headers", headers, ("data", "split", "url", "model", "out")),
                              ("read", read, ("records", "url", "model", "out")),
                              ("judge", judge, ("answers", "reference", "url", "model"))):
        stage = stages.add_parser(name)
        for argument in needed:
            stage.add_argument(f"--{argument}", required=True)
        if name == "headers":
            stage.add_argument("--part", required=True, choices=("dev", "held_out"))
        stage.add_argument("--max-tokens", type=int, default=800)
        stage.add_argument("--effort")
        stage.add_argument("--workers", type=int, default=12)
        stage.set_defaults(run=run)

    arguments = parser.parse_args(argv)
    return arguments.run(arguments)


if __name__ == "__main__":
    sys.exit(main())
