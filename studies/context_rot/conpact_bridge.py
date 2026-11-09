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
import re
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studies.context_rot.bedrock import Converse
from studies.memory_probe.replies import carries_on_the_work

CONDITIONS =("full_with_sc", "full_without_sc", "compaction", "upper_bound")
NOT_SERVED = "[not served: the prompt is longer than the model's window]"


class WindowGuard:
    """An OpenAI client whose calls the server refuses as too long come back as NOT_SERVED.

    The package stops its whole run at one refused call. The server renders a chat about 2%
    longer than the Hugging Face template the window check counts with, so the few prompts
    within 2% of the window are found only when sent. Such a call is answered with NOT_SERVED
    and logged; a call whose messages already hold NOT_SERVED (a probe on an unserved
    summary) is not sent at all. Every other error still stops the run.
    """

    def __init__(self, inner, log: Path | None = None) -> None:
        self._inner, self._log, self._lock = inner, log, threading.Lock()
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if any(NOT_SERVED in str(m.get("content") or "") for m in kwargs.get("messages") or ()):
            return self._not_served(None, "built on an unserved summary")
        import openai

        try:
            return self._inner.chat.completions.create(**kwargs)
        except openai.BadRequestError as error:
            found = re.search(r"Input length \((\d+)\) exceeds model's maximum context length", str(error))
            if not found:
                raise
            return self._not_served(int(found.group(1)), "input longer than the window")

    def _not_served(self, tokens, why):
        if self._log is not None:
            with self._lock, Path(self._log).open("a") as written:
                written.write(json.dumps({"input_tokens": tokens, "why": why}) + "\n")
        message = types.SimpleNamespace(content=NOT_SERVED, role="assistant")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="not_served")],
                                     usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0))


def guarded_endpoint(model: str, url: str, params: dict, log: Path | None):
    """conpact's Endpoint whose client is wrapped in WindowGuard."""
    from conpact import Endpoint

    class Guarded(Endpoint):
        def client(self):
            return WindowGuard(super().client(), log)

    return Guarded(model, url, "EMPTY", params)


class KeepToSummaries:
    """The package's summarizer client for a second model, which may carry on the work in place of summarizing (Laguna S 2.1,
    Nov 6 and 9). The package keeps the whole reply, and so does this, word for word, unless the reply carries on the work
    (a tool call, its markup or the end of the model's thinking), is empty or was cut off by the output cap: then it is asked
    again, up to three tries, and after the last the package's own reply goes back, as the package would have kept it. A reply
    in no particular form is kept as it is. Every call says on the record how many tries it took and whether the reply kept was
    clean. A call the server could not take (NOT_SERVED) goes back at once. gpt-oss-120b's stored summaries pass this in 3,297 of
    3,300 (the three quote the tool-call tag in prose), so its rows and a second model's are read alike."""

    _log_lock = threading.Lock()

    def __init__(self, inner, log: Path | None = None, tries: int = 3) -> None:
        self._inner, self._log, self._tries = inner, log, tries
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        for attempt in range(1, self._tries + 1):
            reply = self._inner.chat.completions.create(**kwargs)
            choice = reply.choices[0]
            if choice.finish_reason == "not_served":
                self._write({"tries": attempt, "clean": False, "not_served": True})
                return reply
            text = choice.message.content or ""
            if choice.finish_reason != "length" and text.strip() and not carries_on_the_work(text):
                self._write({"tries": attempt, "clean": True})
                return reply
        self._write({"tries": self._tries, "clean": False})
        return reply

    def _write(self, row: dict) -> None:
        if self._log is not None:
            with self._log_lock, Path(self._log).open("a") as written:
                written.write(json.dumps(row) + "\n")


def keeping_endpoint(model: str, url: str, params: dict, log: Path | None, keeps: Path):
    """conpact's Endpoint for the summarizer of a second model: WindowGuard, then KeepToSummaries."""
    from conpact import Endpoint

    class Keeping(Endpoint):
        def client(self):
            return KeepToSummaries(WindowGuard(super().client(), log), keeps)

    return Keeping(model, url, "EMPTY", params)


def summary_keep_counts(path: Path) -> dict:
    """What KeepToSummaries put on the record: replies kept clean at the first try, clean after more, and none clean."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []
    return {"calls": len(rows), "clean_first_try": sum(1 for r in rows if r["clean"] and r["tries"] == 1),
            "clean_after_more": sum(1 for r in rows if r["clean"] and r["tries"] > 1),
            "none_clean": sum(1 for r in rows if not r["clean"] and not r.get("not_served")),
            "not_served": sum(1 for r in rows if r.get("not_served"))}


def not_served_counts(results: list[dict]) -> dict:
    """Cases whose summary, or whose probe in each condition, the server could not take."""
    counts = {"summary": sum(any(NOT_SERVED in str(m.get("content")) for m in row.get("compacted") or ())
                             for row in results)}
    for condition in CONDITIONS:
        counts[condition] = sum(row.get(f"{condition}_output") == NOT_SERVED for row in results)
    return counts
JUDGE_MODEL = "global.openai.gpt-5.4"
JUDGE_PRICE = (2.50, 15.00)  # US dollars per million input and output tokens, OpenAI's list price for GPT-5.4


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

    def cost(self) -> float:
        """Spend so far, in US dollars, at ``JUDGE_PRICE``."""
        return (self.input_tokens * JUDGE_PRICE[0] + self.output_tokens * JUDGE_PRICE[1]) / 1e6


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


@contextlib.contextmanager
def reuse_full_history(stored: dict, conditions=("full_with_sc", "full_without_sc")):
    """Answer the two full-history probes from an earlier run of the same cases.

    They do not depend on the summary prompt, so a second prompt's run need not ask them
    again: the package's _probe is answered from ``stored`` (id -> results row) when its
    context is the case's own history, with or without the rule. Anything else is asked.
    """
    evaluate = importlib.import_module("conpact.evaluate")
    original = evaluate._probe

    def _probe(client, endpoint, inst, context, user_turn):
        row = stored.get(inst.id)
        if row is not None and "full_with_sc" in conditions and context is inst.messages:
            return row["full_with_sc_output"], row["full_with_sc_compliant"]
        if row is not None and "full_without_sc" in conditions and context is inst.messages_without_sc:
            return row["full_without_sc_output"], row["full_without_sc_compliant"]
        return original(client, endpoint, inst, context, user_turn)

    evaluate._probe = _probe
    try:
        yield
    finally:
        evaluate._probe = original


def drop_cue(instances):
    """The same cases with the rule stated without "For the rest of this session.".

    The rule is re-rendered by the package's own render_sc (direct=False) and swapped in only
    where it was injected: the turns in sc_message_indices and the restated probe. History,
    probes and the rule-free history stay the same objects, so a no-cue run pairs case by case
    with the published run and can reuse its probe on the rule-free history.
    """
    import dataclasses

    from conpact.constraints import render_sc

    out = []
    for inst in instances:
        plain = render_sc(inst.sc_text, inst.strict, False)
        messages = list(inst.messages)
        for index in inst.sc_message_indices:
            content = messages[index]["content"]
            if not content.startswith(inst.sc_rendered + "\n"):
                raise SystemExit(f"{inst.id}: rule not found where it was injected (message {index})")
            messages[index] = {**messages[index], "content": plain + content[len(inst.sc_rendered):]}
        if not inst.post_sc_probe_prompt.startswith(inst.sc_rendered):
            raise SystemExit(f"{inst.id}: restated probe does not start with the rule")
        out.append(dataclasses.replace(inst, direct=False, sc_rendered=plain, messages=messages,
                                       post_sc_probe_prompt=plain + inst.post_sc_probe_prompt[len(inst.sc_rendered):]))
    return out


def load_instances(source: str, revision: str, length: int | None = None, direct: bool = True,
                   no_cue: bool = False):
    """The paper's static set, or (with ``length``) instances built from its long haystacks.

    Built instances follow the package's own builder (conpact.generate.ConPact); with
    direct=False the rules carry no "For the rest of this session." cue.
    """
    from conpact import load_static

    if no_cue:
        return drop_cue(load_static(source, revision=revision))
    if length is None:
        if not direct:
            raise SystemExit("--no-cue needs --build-length (the static set always carries the cue)")
        return load_static(source, revision=revision)
    from conpact.generate import ConPact

    return ConPact(source=source, length=length, direct=direct, revision=revision).generate()


def judge_deferred(out: Path, judge: RetentionJudge, prompts, workers: int = 8,
                   spend_cap: float | None = None) -> list[dict]:
    """Ask the verdicts a deferred run left empty; keep results.jsonl as written.

    ``prompts(sc_text, compacted)`` is the package's ``judge_prompts``. Rows that already
    hold a verdict are not asked again, nor rows whose verdict an earlier, interrupted
    call wrote to verdicts.jsonl. Once ``judge.cost()`` reaches ``spend_cap`` no new row is
    asked; those stay empty. Writes results.judged.jsonl in the original order.
    """
    out = Path(out)
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    texts = {item["id"]: item["sc_text"]
             for item in map(json.loads, (out / "judge_inputs.jsonl").read_text().splitlines())}
    ledger = out / "verdicts.jsonl"
    earlier = {}
    if ledger.exists():
        earlier = {item["id"]: item["retention"] for item in map(json.loads, ledger.read_text().splitlines())}
    for row in rows:
        if row.get("retention") is None and earlier.get(row["id"]) is not None:
            row["retention"] = earlier[row["id"]]
    todo = [row for row in rows if row.get("retention") is None
            and not any(NOT_SERVED in str(m.get("content")) for m in row.get("compacted") or ())]
    lock = threading.Lock()

    def ask(row: dict) -> None:
        if spend_cap is not None and judge.cost() >= spend_cap:
            return
        row["retention"] = judge.verdict(*prompts(texts[row["id"]], row["compacted"]))
        with lock, ledger.open("a") as written:
            written.write(json.dumps({"id": row["id"], "retention": row["retention"]}) + "\n")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(ask, todo))
    (out / "results.judged.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def over_window(instances, counters: dict, window: int, output_cap: int) -> list[tuple[str, str, int]]:
    """Prompts that do not fit the served window at all, found before any call.

    vLLM refuses only these, and one refused summary stops the package's whole run. A prompt
    that fits but leaves less than ``output_cap`` is served with the cap cut to the room left
    (vLLM 0.32, entrypoints/serve/utils/api_utils.py get_max_tokens); see ``short_room``.
    ``counters`` maps a call kind to a function giving its prompt tokens for an instance.
    """
    over = []
    for row in instances:
        for kind, count in counters.items():
            tokens = count(row)
            if tokens >= window:
                over.append((row.id, kind, tokens))
    return over


def short_room(counts: dict, window: int, output_cap: int) -> list[tuple[str, str, int]]:
    """Calls the server will answer with less than ``output_cap`` tokens of room: (id, kind, room)."""
    return [(case, kind, window - tokens) for case, kinds in counts.items() for kind, tokens in kinds.items()
            if tokens < window and tokens + output_cap > window]


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


def history_range(instances, start: int, end: int):
    """The instances of histories ``start`` to ``end`` (end excluded), counted in file order."""
    order = []
    for row in instances:
        if row.haystack_id not in order:
            order.append(row.haystack_id)
    keep = set(order[start:end])
    return [row for row in instances if row.haystack_id in keep]


def first_contexts(instances, count: int):
    """The instances of the first ``count`` histories, in file order (their configs' test mode)."""
    return history_range(instances, 0, count)


def merge_slices(folders, out: Path, verdicts_from: Path | None = None, expected: int | None = None) -> list[dict]:
    """One run folder from runs over separate history ranges, each case once.

    Verdicts already asked for a case (``verdicts_from``, a judged results file) are kept,
    so judging the merged run asks only the new cases.
    """
    rows, inputs, seen = [], [], set()
    for folder in map(Path, folders):
        for row in map(json.loads, (folder / "results.jsonl").read_text().splitlines()):
            if row["id"] in seen:
                raise SystemExit(f"case {row['id']} is in two slices or twice in {folder}")
            seen.add(row["id"])
            rows.append(row)
        inputs += (folder / "judge_inputs.jsonl").read_text().splitlines()
    if expected is not None and len(rows) != expected:
        raise SystemExit(f"{len(rows)} cases, expected {expected}")
    if verdicts_from is not None:
        known = {item["id"]: item["retention"] for item in map(json.loads, Path(verdicts_from).read_text().splitlines())}
        for row in rows:
            if row.get("retention") is None and known.get(row["id"]) is not None:
                row["retention"] = known[row["id"]]
    kept = {json.loads(line)["id"]: line for line in inputs}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "judge_inputs.jsonl").write_text("".join(kept[row["id"]] + "\n" for row in rows))
    return rows


DATA_REVISION = "9381c98d6fe131a7e622986389800502a9ad055b"  # the Hugging Face release read for this study
OUTPUT_CAP = 4096  # the package's probe and summary setting


def effort_arg(value: str) -> str | None:
    """A command line's reasoning effort: \"none\" sends none. Its own copy, as this runs where the memory code's git library is not."""
    return None if value == "none" else value


def probe_params(effort: str | None = "low", cap: int = OUTPUT_CAP) -> dict:
    """The served model's request settings: the package's own for gpt-oss-120b (effort low, cap 4,096). A model that takes no
    effort is sent none, and one that thinks at length may have a larger cap; a run on other settings records them."""
    return ({"reasoning_effort": effort} if effort is not None else {}) | {"max_tokens": cap}


def window_counters(tokenizer, compaction_messages) -> dict:
    """Prompt-token counters per call kind, from the chat template's rendered text.

    Counting the text, not ``apply_chat_template(tokenize=True)``, because transformers 5
    returns an encoding object there whose length is its number of keys.
    """

    def tokens(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return {"compaction": lambda row: tokens(compaction_messages(row.messages)),
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
              "not_served": not_served_counts(rows),
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
    rows = judge_deferred(out, judge, judge_prompts, workers=arguments.workers, spend_cap=arguments.spend_cap)
    settings = {**_judge_settings(judge), "spent_usd": round(judge.cost(), 4), "spend_cap_usd": arguments.spend_cap,
                "left_unjudged": sum(row.get("retention") is None for row in rows) - judge.unparseable}
    return _report(out, rows, settings, "summary.judged.csv")


def run(arguments) -> dict:
    from conpact import Endpoint, LLMSummarize, evaluate

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    instances = load_instances(arguments.source, arguments.revision, arguments.build_length, not arguments.no_cue,
                               arguments.drop_cue)
    if arguments.histories:
        start, end = (int(part) for part in arguments.histories.split(":"))
        instances = history_range(instances, start, end)
    elif arguments.contexts:
        instances = first_contexts(instances, arguments.contexts)
    if arguments.variant:
        from studies.memory_probe.variants import apply

        instances = apply(instances, arguments.variant)
    if arguments.tokenizer:
        from conpact.prompts import summarization_prompt
        from transformers import AutoTokenizer

        counters = window_counters(AutoTokenizer.from_pretrained(arguments.tokenizer),
                                   lambda messages: summarization_prompt(arguments.prompt, messages))
        # A summary or probe this small means the count is broken, not that the prompt fits.
        if min(each for row in instances[:1] for each in (count(row) for count in counters.values())) < 1000:
            raise SystemExit("prompt token count is implausibly small; the window check is not working")
        counts = {row.id: {kind: count(row) for kind, count in counters.items()} for row in instances}
        over = over_window(instances, {kind: (lambda row, kind=kind: counts[row.id][kind]) for kind in counters},
                           arguments.window, arguments.output_cap)
        largest = {kind: max(each[kind] for each in counts.values()) for kind in counters}
        short = short_room(counts, arguments.window, arguments.output_cap)
        (out / "window.json").write_text(json.dumps({"window": arguments.window, "output_cap": arguments.output_cap,
                                                     "largest_prompt_tokens": largest, "over": over,
                                                     "short_room": short, "prompt_tokens": counts}, indent=1))
        if over:
            raise SystemExit(f"{len(over)} prompts do not fit the {arguments.window}-token window; "
                             f"see {out / 'window.json'}")
        if arguments.check_only:
            return {"instances": len(instances), "largest_prompt_tokens": largest, "over": over,
                    "short_room_calls": len(short), "smallest_room": min((r for *_, r in short), default=None)}
    (out / "judge_inputs.jsonl").write_text(
        "".join(json.dumps({"id": row.id, "sc_text": row.sc_text}) + "\n" for row in instances))
    params = probe_params(effort_arg(arguments.effort), arguments.output_cap)
    if params != probe_params():
        (out / "probe.json").write_text(json.dumps({"model": arguments.probe_model, "params": params}, indent=1))
    oss = guarded_endpoint(arguments.probe_model, arguments.probe_url, params, out / "not_served.jsonl")
    writer = oss
    if arguments.keep_to_summaries:
        writer = keeping_endpoint(arguments.probe_model, arguments.probe_url, params, out / "not_served.jsonl", out / "summary_keeps.jsonl")
    summarizer = LLMSummarize(writer, prompt=arguments.prompt)
    unused = Endpoint("bedrock-judge", api_key="unused")
    reuse = contextlib.nullcontext()
    if arguments.reuse_full:
        stored = {row["id"]: row for row in map(json.loads,
                                                 (Path(arguments.reuse_full) / "results.jsonl").read_text().splitlines())}
        reuse = reuse_full_history(stored, ("full_without_sc",) if arguments.reuse_without_only
                                   else ("full_with_sc", "full_without_sc"))
    if arguments.judge == "defer":
        with reuse, deferred_judge():
            evaluate(instances, summarizer, out_dir=out, probe=oss, judge=unused,
                     compact_workers=arguments.workers, workers=arguments.workers)
        settings = {"model": JUDGE_MODEL, "deferred": True}
    else:
        judge = RetentionJudge(Converse(JUDGE_MODEL, effort="low", max_tokens=arguments.judge_max_tokens),
                               log=out / "judge_calls.jsonl")
        with reuse, bedrock_judge(judge):
            evaluate(instances, summarizer, out_dir=out, probe=oss, judge=unused,
                     compact_workers=arguments.workers, workers=arguments.workers)
        settings = _judge_settings(judge)
    ids = {row.id for row in instances}
    results = [r for r in map(json.loads, (out / "results.jsonl").read_text().splitlines()) if r["id"] in ids]
    report = _report(out, results, settings, "summary.csv")
    if arguments.keep_to_summaries:
        report["summary_keeps"] = summary_keep_counts(out / "summary_keeps.jsonl")
        (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="hermes", choices=("hermes", "openresearcher", "wildchat",
                                                                 "swe_natural"))
    parser.add_argument("--contexts", type=int, default=5, help="first N histories x 15 rules; 0 for all 50")
    parser.add_argument("--histories", help="START:END, histories in file order, end excluded (overrides --contexts)")
    parser.add_argument("--build-length", type=int, help="build instances of this many tokens from the long haystacks")
    parser.add_argument("--no-cue", action="store_true", help="with --build-length: rules without the session cue")
    parser.add_argument("--variant", help="middle, bottom, revised, withdrawn or cued (studies.memory_probe.variants)")
    parser.add_argument("--drop-cue", action="store_true",
                        help="the published cases with the rule's session cue removed (paired with the published run)")
    parser.add_argument("--reuse-without-only", action="store_true",
                        help="with --reuse-full: reuse only the rule-free history probe (the rule text changed)")
    parser.add_argument("--reuse-full", help="results folder of an earlier run of the same cases whose full-history "
                                             "probes are reused (another summary prompt)")
    parser.add_argument("--prompt", default="pi_mono", choices=("pi_mono", "anthropic"))
    parser.add_argument("--probe-url", help="the served summarizer and probe model; not needed with --judge-later")
    parser.add_argument("--probe-model", default="openai/gpt-oss-120b")
    parser.add_argument("--effort", default="low", help='the probe model\'s reasoning effort; "none" sends none')
    parser.add_argument("--output-cap", type=int, default=OUTPUT_CAP, help="the probe model's output cap, summaries included")
    parser.add_argument("--keep-to-summaries", action="store_true",
                        help="ask again a summary reply that carries on the work, is empty or was cut off, up to three tries (a second model's summaries)")
    parser.add_argument("--revision", default=DATA_REVISION, help="Hugging Face revision of the instances")
    parser.add_argument("--judge", default="bedrock", choices=("bedrock", "defer"),
                        help="defer: leave verdicts empty, for --judge-later where the judge key is")
    parser.add_argument("--judge-later", action="store_true", help="ask the verdicts a deferred run in --out left")
    parser.add_argument("--tokenizer", help="model folder; check every prompt fits the window before any call")
    parser.add_argument("--window", type=int, default=131072)
    parser.add_argument("--check-only", action="store_true", help="with --tokenizer: check the window, call nothing")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--spend-cap", type=float, help="with --judge-later: stop asking once this many dollars spent")
    parser.add_argument("--merge", nargs="+", help="run folders over separate history ranges, merged into --out")
    parser.add_argument("--expected", type=int, help="with --merge: the number of cases the merged run must hold")
    parser.add_argument("--verdicts-from", help="with --merge: a judged results file whose verdicts are kept")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    if arguments.merge:
        rows = merge_slices(arguments.merge, Path(arguments.out), arguments.verdicts_from, arguments.expected)
        print(json.dumps({"cases": len(rows), "judged_already": sum(row.get("retention") is not None for row in rows)}))
        return 0
    if not (arguments.judge_later or arguments.check_only) and not arguments.probe_url:
        parser.error("--probe-url is needed unless --judge-later or --check-only")
    if arguments.check_only and not arguments.tokenizer:
        parser.error("--check-only needs --tokenizer")
    print(json.dumps(judge_later(arguments) if arguments.judge_later else run(arguments), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
