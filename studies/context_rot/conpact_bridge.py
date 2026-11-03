"""Run the rule-loss benchmark (conpact, "Lost in Context Compaction") with a Bedrock judge.

Everything is the package's own (conpact 0.1.0, repo commit e38e298): the instances
from Hugging Face, the compactors and their prompts, the probe prompts and its four
cases, the judge's prompt (``conpact.prompts.judge_prompts``) and its result file.
Only the retention judge's transport changes: their GPT-5.4 is called through OpenAI,
which this study cannot use, so the same model is asked through Amazon Bedrock. Their
rule is kept: a verdict is the reply, stripped and upper-cased, starting with YES or NO,
asked up to three times. Where their code then raises and stops the run, this records
the verdict as unknown and counts it.

Also reported, because their summary drops them silently: how many probe answers were
not a single A or B in each case, how many retention verdicts were unclear, and the
compliance under the paper's exact grading rule (the whole reply must be one letter)
beside the package's (the last non-empty line).

    python -m studies.context_rot.conpact_bridge --source hermes --contexts 5 \
        --prompt pi_mono --probe-url http://<gpu-node>:8000/v1 --out runs/conpact/hermes-pi

On a GPU cluster that must not hold the judge key, run with ``--judge defer`` there, copy
the output folder back, and ask the verdicts with ``--judge-later --out <folder>``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studies.context_rot.bedrock import Converse

CONDITIONS = ("full_with_sc", "full_without_sc", "compaction", "upper_bound")
JUDGE_MODEL = "global.openai.gpt-5.4"


class RetentionJudge:
    """Their retention judge's rule, asked of a Bedrock model."""

    def __init__(self, converse: Converse, tries: int = 3, log: Path | None = None) -> None:
        self.converse = converse
        self.tries = tries
        self.log = log
        self.calls = self.unparseable = self.input_tokens = self.output_tokens = 0
        self._lock = threading.Lock()

    def verdict(self, system: str, user: str) -> bool | None:
        for _ in range(self.tries):
            reply = self.converse.ask(user, system=system)
            text = reply.text.strip().upper()
            with self._lock:
                self.calls += 1
                self.input_tokens += reply.input_tokens
                self.output_tokens += reply.output_tokens
                if self.log is not None:
                    with self.log.open("a") as written:
                        written.write(json.dumps({"output": reply.text, "input_tokens": reply.input_tokens,
                                                  "output_tokens": reply.output_tokens,
                                                  "stop_reason": reply.stop_reason}) + "\n")
            if text.startswith("YES"):
                return True
            if text.startswith("NO"):
                return False
        with self._lock:
            self.unparseable += 1
        return None


@contextlib.contextmanager
def bedrock_judge(judge: RetentionJudge):
    """Put ``judge`` in the place of conpact's own judge call while the block runs."""
    # The package's __init__ re-exports a function named evaluate over this submodule.
    evaluate = importlib.import_module("conpact.evaluate")
    original = evaluate._judge

    def _judge(client, endpoint, sc_text, compacted):
        return judge.verdict(*evaluate.judge_prompts(sc_text, compacted))

    evaluate._judge = _judge
    try:
        yield judge
    finally:
        evaluate._judge = original


@contextlib.contextmanager
def deferred_judge():
    """Leave every retention verdict empty, to be asked later by ``judge_deferred``.

    The GPU cluster runs the summaries and probes without any judge key; the verdicts
    are then asked from the workstation that holds the key, on the same summaries.
    """
    evaluate = importlib.import_module("conpact.evaluate")
    original = evaluate._judge
    evaluate._judge = lambda client, endpoint, sc_text, compacted: None
    try:
        yield
    finally:
        evaluate._judge = original


def judge_deferred(out: Path, judge: RetentionJudge, prompts, workers: int = 8) -> list[dict]:
    """Ask the verdicts a deferred run left empty; keep results.jsonl as written.

    ``prompts(sc_text, compacted)`` is the package's ``judge_prompts``. Rows that already
    hold a verdict are not asked again. Writes results.judged.jsonl in the original order.
    """
    out = Path(out)
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    texts = {item["id"]: item["sc_text"]
             for item in map(json.loads, (out / "judge_inputs.jsonl").read_text().splitlines())}
    todo = [row for row in rows if row.get("retention") is None]

    def ask(row: dict) -> None:
        row["retention"] = judge.verdict(*prompts(texts[row["id"]], row["compacted"]))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(ask, todo))
    (out / "results.judged.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def over_window(instances, counters: dict, window: int, output_cap: int) -> list[tuple[str, str, int]]:
    """Prompts whose tokens plus the output cap pass the served window, found before any call.

    vLLM refuses such a request, and one refused summary stops the package's whole run.
    ``counters`` maps a call kind to a function giving its prompt tokens for an instance.
    """
    over = []
    for row in instances:
        for kind, count in counters.items():
            tokens = count(row)
            if tokens + output_cap > window:
                over.append((row.id, kind, tokens))
    return over


def paper_grade(output: str, compliant_letter: str) -> bool | None:
    """The paper runner's rule (scripts/evaluation.py:458-462): the whole reply is one letter."""
    cleaned = (output or "").strip().upper()
    return None if cleaned not in ("A", "B") else cleaned == compliant_letter


def unparseable_counts(results: list[dict]) -> dict:
    counts = {"retention": sum(row.get("retention") is None for row in results)}
    for condition in CONDITIONS:
        counts[condition] = sum(f"{condition}_compliant" in row and row[f"{condition}_compliant"] is None
                                for row in results)
    counts["instances"] = len(results)
    return counts


def paper_rule_rates(results: list[dict]) -> dict:
    """Compliance per case under the paper's exact grading, with its own unparseable count."""
    out = {}
    for condition in CONDITIONS:
        graded = [paper_grade(row.get(f"{condition}_output", ""), row["compliant_letter"]) for row in results
                  if f"{condition}_output" in row]
        valid = [g for g in graded if g is not None]
        out[condition] = {"rate_pct": 100 * sum(valid) / len(valid) if valid else None,
                          "unparseable": len(graded) - len(valid)}
    return out


def first_contexts(instances, count: int):
    """The instances of the first ``count`` histories, in file order (their configs' test mode)."""
    order = []
    for row in instances:
        if row.haystack_id not in order:
            order.append(row.haystack_id)
    keep = set(order[:count])
    return [row for row in instances if row.haystack_id in keep]


DATA_REVISION = "9381c98d6fe131a7e622986389800502a9ad055b"  # the Hugging Face release read for this study
OUTPUT_CAP = 4096  # the package's probe and summary setting


def _window_counters(tokenizer_path: str, prompt: str) -> dict:
    from conpact.prompts import summarization_prompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokens(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    return {"compaction": lambda row: tokens(summarization_prompt(prompt, row.messages)),
            "full_with_sc": lambda row: tokens([{"role": "system", "content": row.system_prompt},
                                                {"role": "developer", "content": row.developer_prompt},
                                                *row.messages, {"role": "user", "content": row.probe_prompt}]),
            "full_without_sc": lambda row: tokens([{"role": "system", "content": row.system_prompt},
                                                   {"role": "developer", "content": row.developer_prompt},
                                                   *row.messages_without_sc,
                                                   {"role": "user", "content": row.probe_prompt}])}


def _report(out: Path, rows: list[dict], judge: dict, summary_file: str) -> dict:
    from conpact.evaluate import summarize

    summary = summarize(rows)
    with (out / summary_file).open("w", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = {"instances": len(rows), "package_summary": summary, "unparseable": unparseable_counts(rows),
              "paper_rule": paper_rule_rates(rows), "judge": judge}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def _judge_settings(judge: RetentionJudge) -> dict:
    return {**judge.converse.settings(), "calls": judge.calls, "unclear": judge.unparseable,
            "input_tokens": judge.input_tokens, "output_tokens": judge.output_tokens}


def judge_later(arguments) -> dict:
    from conpact.prompts import judge_prompts

    out = Path(arguments.out)
    judge = RetentionJudge(Converse(JUDGE_MODEL, effort="low", max_tokens=arguments.judge_max_tokens),
                           log=out / "judge_calls.jsonl")
    rows = judge_deferred(out, judge, judge_prompts, workers=arguments.workers)
    return _report(out, rows, _judge_settings(judge), "summary.judged.csv")


def run(arguments) -> dict:
    from conpact import Endpoint, LLMSummarize, evaluate, load_static

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    instances = load_static(arguments.source, revision=arguments.revision)
    if arguments.contexts:
        instances = first_contexts(instances, arguments.contexts)
    if arguments.tokenizer:
        counters = _window_counters(arguments.tokenizer, arguments.prompt)
        counts = {row.id: {kind: count(row) for kind, count in counters.items()} for row in instances}
        over = over_window(instances, {kind: (lambda row, kind=kind: counts[row.id][kind]) for kind in counters},
                           arguments.window, OUTPUT_CAP)
        largest = {kind: max(each[kind] for each in counts.values()) for kind in counters}
        (out / "window.json").write_text(json.dumps({"window": arguments.window, "output_cap": OUTPUT_CAP,
                                                     "largest_prompt_tokens": largest, "over": over,
                                                     "prompt_tokens": counts}, indent=1))
        if over:
            raise SystemExit(f"{len(over)} prompts pass the {arguments.window}-token window with a "
                             f"{OUTPUT_CAP}-token cap; see {out / 'window.json'}")
        if arguments.check_only:
            return {"instances": len(instances), "largest_prompt_tokens": largest, "over": over}
    (out / "judge_inputs.jsonl").write_text(
        "".join(json.dumps({"id": row.id, "sc_text": row.sc_text}) + "\n" for row in instances))
    oss = Endpoint(arguments.probe_model, arguments.probe_url, "EMPTY",
                   {"reasoning_effort": "low", "max_tokens": OUTPUT_CAP})
    summarizer = LLMSummarize(oss, prompt=arguments.prompt)
    unused = Endpoint("bedrock-judge", api_key="unused")
    if arguments.judge == "defer":
        with deferred_judge():
            evaluate(instances, summarizer, out_dir=out, probe=oss, judge=unused,
                     compact_workers=arguments.workers, workers=arguments.workers)
        settings = {"model": JUDGE_MODEL, "deferred": True}
    else:
        judge = RetentionJudge(Converse(JUDGE_MODEL, effort="low", max_tokens=arguments.judge_max_tokens),
                               log=out / "judge_calls.jsonl")
        with bedrock_judge(judge):
            evaluate(instances, summarizer, out_dir=out, probe=oss, judge=unused,
                     compact_workers=arguments.workers, workers=arguments.workers)
        settings = _judge_settings(judge)
    ids = {row.id for row in instances}
    results = [r for r in map(json.loads, (out / "results.jsonl").read_text().splitlines()) if r["id"] in ids]
    return _report(out, results, settings, "summary.csv")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="hermes", choices=("hermes", "openresearcher", "wildchat",
                                                                 "swe_natural"))
    parser.add_argument("--contexts", type=int, default=5, help="first N histories x 15 rules; 0 for all 50")
    parser.add_argument("--prompt", default="pi_mono", choices=("pi_mono", "anthropic"))
    parser.add_argument("--probe-url", help="the served summarizer and probe model; not needed with --judge-later")
    parser.add_argument("--probe-model", default="openai/gpt-oss-120b")
    parser.add_argument("--revision", default=DATA_REVISION, help="Hugging Face revision of the instances")
    parser.add_argument("--judge", default="bedrock", choices=("bedrock", "defer"),
                        help="defer: leave verdicts empty, for --judge-later where the judge key is")
    parser.add_argument("--judge-later", action="store_true", help="ask the verdicts a deferred run in --out left")
    parser.add_argument("--tokenizer", help="model folder; check every prompt fits the window before any call")
    parser.add_argument("--window", type=int, default=131072)
    parser.add_argument("--check-only", action="store_true", help="with --tokenizer: check the window, call nothing")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    if not (arguments.judge_later or arguments.check_only) and not arguments.probe_url:
        parser.error("--probe-url is needed unless --judge-later or --check-only")
    if arguments.check_only and not arguments.tokenizer:
        parser.error("--check-only needs --tokenizer")
    print(json.dumps(judge_later(arguments) if arguments.judge_later else run(arguments), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
