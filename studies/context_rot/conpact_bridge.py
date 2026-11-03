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
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import sys
import threading
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


def run(arguments) -> dict:
    from conpact import Endpoint, LLMSummarize, evaluate, load_static

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    instances = load_static(arguments.source)
    if arguments.contexts:
        instances = first_contexts(instances, arguments.contexts)
    oss = Endpoint(arguments.probe_model, arguments.probe_url, "EMPTY",
                   {"reasoning_effort": "low", "max_tokens": 4096})
    judge = RetentionJudge(Converse(JUDGE_MODEL, effort="low", max_tokens=arguments.judge_max_tokens),
                           log=out / "judge_calls.jsonl")
    with bedrock_judge(judge):
        evaluate(instances, LLMSummarize(oss, prompt=arguments.prompt), out_dir=out, probe=oss,
                 judge=Endpoint("bedrock-judge", api_key="unused"),
                 compact_workers=arguments.workers, workers=arguments.workers)
    ids = {row.id for row in instances}
    results = [r for r in map(json.loads, (out / "results.jsonl").read_text().splitlines()) if r["id"] in ids]
    with (out / "summary.csv").open() as read:
        summary = list(csv.DictReader(read))
    report = {"instances": len(results), "package_summary": summary, "unparseable": unparseable_counts(results),
              "paper_rule": paper_rule_rates(results),
              "judge": {**judge.converse.settings(), "calls": judge.calls, "unclear": judge.unparseable,
                        "input_tokens": judge.input_tokens, "output_tokens": judge.output_tokens}}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="hermes", choices=("hermes", "openresearcher", "wildchat",
                                                                 "swe_natural"))
    parser.add_argument("--contexts", type=int, default=5, help="first N histories x 15 rules; 0 for all 50")
    parser.add_argument("--prompt", default="pi_mono", choices=("pi_mono", "anthropic"))
    parser.add_argument("--probe-url", required=True)
    parser.add_argument("--probe-model", default="openai/gpt-oss-120b")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    print(json.dumps(run(parser.parse_args(argv)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
