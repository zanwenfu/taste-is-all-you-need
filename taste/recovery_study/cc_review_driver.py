"""The Claude Code reviewer over many base runs: one review per run, a few at a time, resumable.

Given a manifest of base runs whose rebuilt final states are in running containers, this calls the reviewer
(``taste.recovery_study.cc_reviewer``) once per run and appends one record per finished review to a results file,
so that a stopped job picks up where it left off. Each record carries the checker's submission (``reply``, schema
``taste.agents/CheckerVerdict/1``, the same object a checker trial's summary holds under ``checker``) and the reviewer's
label (``reviewer``: family ``claude``, model, route, prompts, limits).

The manifest is JSON lines, one per run::

    {"run": "<base run>", "container": "<id or name>", "cwd": "/app",
     "task_text": "<the checker's task text>"            # or "task_file": "<path to it>", or the parts:
     "task": "...", "final_message": "...", "submission": "...", "changed_paths": [...],
     "requirement": "<optional: one requirement, as requirement_task names it>", "user": "<optional exec user>"}

A usage, spend or rate limit, a login failure, or a tool call the hook never saw (``StopJob``) ends the whole job:
no new review starts, those running finish, ``stop.json`` says why, and nothing is retried. A run whose review failed
for another reason is recorded with its error and left for ``retry_failed``; after ``max_failures`` such failures in a
row the job stops too.
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from taste.agents.checker import checker_task
from taste.recovery_study.cc_reviewer import ReviewFailed, StopJob
from taste.recovery_study.requirement_driver import requirement_task

KIND = "taste.cc_reviews/1"


def now():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_manifest(path):
    items, seen = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for name in ("run", "container", "cwd"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ValueError(f"manifest line {number}: {name} must be text")
        if item["run"] in seen:
            raise ValueError(f"manifest line {number}: run {item['run']} is listed twice")
        if not any(key in item for key in ("task_text", "task_file", "task")):
            raise ValueError(f"manifest line {number}: give task_text, task_file or task")
        seen.add(item["run"])
        items.append(item)
    return items


def task_text(item):
    """The checker's task text for one run, as the checker gets it."""
    if "task_text" in item:
        text = item["task_text"]
    elif "task_file" in item:
        text = Path(item["task_file"]).read_text()
    else:
        text = checker_task(item["task"], item.get("final_message", ""), item.get("submission", ""),
                            changed_paths=item.get("changed_paths"))
    if item.get("requirement"):
        text = requirement_task(text, item["requirement"])
    return text


def read_results(path):
    """The records written so far, by run (the last record of a run wins); a torn last line is skipped."""
    found = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("kind") == KIND and isinstance(record.get("run"), str):
                found[record["run"]] = record
    return found


def _slug(run_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)[:120]


class Writer:
    """Appends whole records, one line each, flushed to disk before the next review is counted done."""

    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def drive(items, results, work_root, runner, *, concurrency=2, retry_failed=False, limit=None, max_failures=3,
          log=print):
    """Review every run of ``items`` without a record in ``results``; returns a summary of this pass."""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a whole number, 1 or more")
    done = read_results(results)
    todo = [item for item in items
            if item["run"] not in done or (retry_failed and done[item["run"]].get("reply") is None)]
    already = len(items) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    identity = runner.identity(runner.version())
    writer, stop, failures = Writer(results), {}, {"row": 0}
    counts = {"reviewed": 0, "failed": 0, "already_recorded": already}
    lock = threading.Lock()

    def one(item):
        started = now()
        base = {"kind": KIND, "run": item["run"], "reviewer": identity, "container": item["container"],
                "cwd": item["cwd"], "started": started}
        try:
            review = runner.review(item["container"], item["cwd"], task_text(item),
                                   Path(work_root) / _slug(item["run"]), user=item.get("user"))
        except ReviewFailed as error:
            writer.write({**base, "reply": None, "error": str(error)[:2000], "ended": now()})
            with lock:
                counts["failed"] += 1
                failures["row"] += 1
                if failures["row"] >= max_failures and not stop:
                    stop.update(kind="failures", message=f"{failures['row']} reviews failed in a row; last: "
                                                         f"{str(error)[:500]}", run=item["run"], at=now())
            log(f"{item['run']}: failed: {str(error)[:200]}")
            return
        writer.write({**base, "reply": review.reply, "claude_code": review.claude_code, "hook": review.hook,
                      "ended": now()})
        with lock:
            counts["reviewed"] += 1
            failures["row"] = 0
        reply = review.reply
        log(f"{item['run']}: {reply['verdict']} (ended {reply['ended']}, steps {reply['steps']}, "
            f"commands {reply['commands']}, refused {reply['refused']})")

    def guarded(item):
        try:
            one(item)
        except StopJob as error:
            with lock:
                if not stop:
                    stop.update(kind=error.kind, message=error.message[:2000], run=item["run"], at=now())
            log(f"{item['run']}: STOP ({error.kind}); no new review starts")

    pending = list(todo)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        running = set()
        while pending or running:
            while pending and not stop and len(running) < concurrency:
                running.add(pool.submit(guarded, pending.pop(0)))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
    summary = {**counts, "left": len(pending), "stopped": dict(stop) or None, "results": str(results)}
    if stop:
        stop_path = Path(results).with_name("stop.json")
        stop_path.write_text(json.dumps({**stop, "left": len(pending)}, indent=1) + "\n")
        summary["stop_file"] = str(stop_path)
    return summary


def verdicts(results):
    """The reviews by verdict: done, not_done, none (a limit ended the review), failed (no review)."""
    tally = {"done": 0, "not_done": 0, "none": 0, "failed": 0}
    for record in read_results(results).values():
        reply = record.get("reply")
        if reply is None:
            tally["failed"] += 1
        else:
            tally[reply["verdict"] or "none"] += 1
    return tally
