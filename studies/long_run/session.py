"""One session of the long run: a chain of real tasks in one conversation, with compaction on.

A session file names the tasks and the rules to plant. A rule is told with the first task or as a user message at a
chosen model call, and never again, except in the ceiling arm that restates it. Each rule carries the patterns that
score it from the command record.

    compaction     the summary alone: a baseline
    taste          the same, and every message committed to Taste's memory, with search and fetch offered beside bash
    taste_exists   the same, and the system prompt says a searchable record exists
    taste_check    the same, and it also says to check that record before answering a request
    taste_search   compaction, every message committed to Taste's memory, and at each new task the harness itself searches it with the task's own
                   words and puts what comes back in the system prompt until the next task; no tools, so the agent's own wish to look plays no part
    taste_rules    compaction, each user message read once as it arrives, what it asks to keep doing written in that commit's
                   header, and all of that read back from the store and put in the system prompt at every call; no tools
    notes_told     compaction, and a notes file a model rewrites at each compaction from what is about to be folded away, told to keep
                   what the user asked it to keep doing; every version in Taste's memory, the newest in the system prompt at every call
    notes_neutral  the same, not told that
    mem0_search    compaction, every message the agent records added to Mem0 (studies/long_run/mem0_store.py), and at each new task the
                   harness searches Mem0 with the task's own words and puts what comes back in the system prompt, as taste_search does
    no_compaction  a ceiling: the whole conversation, as long as it fits
    restated       a ceiling: compaction, with every rule told so far in the system prompt at every call

Names, per the study's rule: "taste" is "Taste memory, study-driven"; the others have no Taste code in the measured path.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studies.context_rot.repeated import round_prompt
from studies.long_run.agent import LongRunAgent, MemoryEnvironment, Stopped, ToolModel, render
from studies.long_run.score import score_session
from studies.memory_bench.core import fill
from studies.memory_ledger import BASELINE, ClaimRefused, check, stand_in_record
from studies.memory_probe.cases import (
    DELIVERED_HEADER,
    FOUND_HEADER,
    LOOK,
    NOTES_HEADER,
    SENTENCE,
    keep_newest,
    search_with,
)
from studies.memory_probe.notes import _NO_NOTES
from studies.memory_probe.notes import NOTES_LAST as _NOTES_LAST
from studies.memory_probe.replies import carries_on_the_work as _carries_on_the_work
from studies.memory_probe.replies import read_summary
from studies.memory_probe.stores import GitTurns, NotesStore
from taste.memstore import CommitHeader

TOOL_ARMS = ("taste", "taste_exists", "taste_check")
# What the harness's search brings is the words of earlier turns, and a rule told with a task brings that task along. On the first Laguna session, under the matrix's header, the agent took
# the old task for its own in two tasks of five, 30 of its 201 commands against none of 1,527 in the other arms, even with a right summary before it ("I see the task is about creating a logic-gate simulator"). So a harness says what the block is.
SEARCH_HEADER = ("Earlier turns from the record of this session that may bear on the current task, oldest first. Where two disagree, the later one stands. "
                 "They are history and not new requests: any task in them is over, unless the current task says otherwise.")
SEARCH_ARMS = {"taste_search": SEARCH_HEADER, "taste_search_bare": FOUND_HEADER}
NOTE_ARMS = {"notes_told": "told", "notes_neutral": "neutral"}
MEM0_ARMS = {"mem0_search": SEARCH_HEADER}
ARMS = ("compaction", *TOOL_ARMS, *SEARCH_ARMS, "taste_rules", *NOTE_ARMS, "no_compaction", "restated", *MEM0_ARMS)
MEMORY_BUDGET = 4096
EFFORT = "low"  # the reasoning effort on every model call of a session: the agent's, every writer's, and Mem0's
MEM0_THINKING = "user"  # Mem0's model thinks on exchanges holding a user message, as Taste's rule reader reads them; not on command output (mem0_sidecar)
MEM0_MAX_TOKENS = 4096  # Mem0's output cap without thinking, as in the chat comparison (memory_bench/swe.py MEM0_INGEST)
MEM0_THINKING_MAX_TOKENS = 16384  # with thinking: over twice the longest reply that ended before the full run (6,214); one cut off is asked again without thinking (mem0_sidecar.Calls)
_SAID_ABOUT_THE_RECORD = {"taste_exists": f" {SENTENCE}", "taste_check": f" {SENTENCE} {LOOK}"}
_SYSTEM = "You are a coding agent. Work on each task until it is done, then run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def load_session(raw: dict) -> dict:
    """Refuse a session that could not be scored before anything runs."""
    if not raw.get("tasks"):
        raise ValueError("a session needs tasks")
    for rule in raw.get("rules", []):
        for key in ("name", "text", "trigger", "obeys"):
            if key not in rule:
                raise ValueError(f"a rule needs a {key}: {rule}")
        if rule.get("say") != "first" and not isinstance(rule.get("say"), int):
            raise ValueError(f'a rule needs say: "first" or a model call: {rule}')
        re.compile(rule["trigger"])
        re.compile(rule["obeys"])
    return {"rules": [], "system_template": _SYSTEM, "instance_template": "{{task}}", "observation_template": "{{output.output}}",
            "format_error_template": "{{error}}", "step_limit": 0, "task_calls": 0, "stop_at_tokens": 0, **raw}


# A model that thinks in its reply, with a long conversation pasted in, often carries on the work after a summary or in place of one. Of the 161 summaries of the first Laguna session 39 had
# other text around the block, and 7 were no summary at all, and the next call had no task in front of it. So a summary is what the published prompt asked for, and nothing around it.
# The reader itself is in studies.memory_probe.replies (moved Nov 9, unchanged), so that the matrix reads a second model's replies by the same rule.
def make_summarizer(chat: Callable, option: str, package_prompt: Callable, *, tries: int = 3, on_reply: Callable[[dict], None] = lambda row: None) -> Callable:
    """The published summary prompts, each later round given the previous summary. A reply empty, cut off or with no summary in it is asked again, and none after the last try gives None."""

    def summarize(older: list[dict], previous: str | None) -> str | None:
        messages = round_prompt(option, previous, older, package_prompt)
        for attempt in range(1, tries + 1):
            choice = chat(messages, None).choices[0]
            text = choice.message.content or ""
            summary = None if choice.finish_reason == "length" else read_summary(option, text)
            on_reply({"try": attempt, "finish": choice.finish_reason, "summary": summary is not None, "reply": text})
            if summary:
                return summary
        return None

    return summarize


# A reading cut off by the output cap says nothing whole, and once ended its arm. Two of the first three main sessions lost the arm that way (Nov 9, seed 130
# about task 4, seed 140 at task 27), so from Nov 9 a cut-off reading is asked again the same way, as a summary is, and every cut-off try is counted.
# The reader's prompt is the instrument and stays as it was; a reading still cut off on the last try ends the arm as before.
RULE_READING_TRIES = 3


def make_reader(chat: Callable, *, tries: int = RULE_READING_TRIES, cut_off: list[int] | None = None) -> tuple[Callable[[str], list[str]], list[int]]:
    """The study's reader, asked once per distinct user message, as it arrives, and again up to `tries` times in all when a reply is cut off."""
    from studies.context_rot.criteria import parse_reply, reader_prompt

    known: dict[str, list[str]] = {}
    calls = [0]
    cut = cut_off if cut_off is not None else [0]

    def read(text: str) -> list[str]:
        if text not in known:
            for _ in range(tries):
                reply = chat(reader_prompt(text), None)
                calls[0] += 1
                if reply.choices[0].finish_reason != "length":
                    known[text] = parse_reply(reply.choices[0].message.content or "", text)[0]
                    break
                cut[0] += 1
            else:
                raise RuntimeError(f"a rule reading was cut off by the output cap on all {tries} tries")
        return known[text]

    return read, calls


# A long conversation pasted in makes a model that thinks first carry on the work: on the first real sessions of Laguna S 2.1, 6 of 12 versions held tool calls. So the instruction comes
# again last, as a summary prompt's does, and a reply that is still not notes is not taken for them. Read on real parts of a session before any main run: 0 of 12 with tool text, against 1 of 12.
# The sentence and the placeholder live with the matrix's note-taker since Nov 9, unchanged, so that both read a model's notes alike.


def make_note_taker(chat: Callable, style: str, store: NotesStore, skipped: list[int]) -> Callable[[list[dict]], None]:
    """The study's note-taker: the whole file rewritten from what a compaction is about to fold away."""
    from studies.memory_probe.notes import _PROMPT, MAX_TURN_CHARS, STYLES

    def keep(older: list[dict]) -> None:
        part = "\n".join(f"[{turn['role']}]: {str(turn['content'])[:MAX_TURN_CHARS]}" for turn in older)
        wanted = _PROMPT.format(told=STYLES[style], notes=store.newest() or _NO_NOTES, part=part) + _NOTES_LAST
        for _ in range(EMPTY_TRIES):
            reply = chat([{"role": "user", "content": wanted}], None)
            text = reply.choices[0].message.content or ""
            if text.strip() and text.strip() != _NO_NOTES and not _carries_on_the_work(text) and reply.choices[0].finish_reason != "length":
                store.add_version(text)
                return
        # An empty reply, a cut-off one or one that is not notes says nothing whole about them: the file stays as it was, rather than being wiped or halved, and the count is on the record.
        skipped[0] += 1

    return keep


EMPTY_TRIES = 3
THIN_FOLD = 1000  # tokens a compaction must take off for it to count as one that folded anything much: a very long summary can leave a run compacting at every turn


def _counted(chat: Callable, used: dict[str, int]) -> Callable:
    """Every model call a session makes, and its tokens, whoever asked: the agent, a summary, the reader or the note-taker."""

    def counting(messages: list[dict], tools: list | None) -> Any:
        reply = chat(messages, tools)
        usage = getattr(reply, "usage", None)
        used["calls"] += 1
        used["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        used["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return reply

    return counting


def run_session(session: dict, *, arm: str, chat: Callable, make_env: Callable[[dict], Any], count: Callable[[str], int],
                out: Path, workdir: Path, package_prompt: Callable, versions: dict | None = None, disarm: Callable[[], None] = lambda: None,
                mem0: Callable[[], Any] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm in MEM0_ARMS and mem0 is None:
        raise ValueError(f"the {arm} arm needs a Mem0 client")
    out, workdir = Path(out), Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    rules = session["rules"]
    model_use = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    notes_skipped = [0]
    chat = _counted(chat, model_use)
    store: Any = None
    standing, on_compact, reader_calls, reader_cut = None, None, [0], [0]
    searches: list[dict] = []
    found = [""]
    if arm in TOOL_ARMS:
        store = GitTurns(workdir)
    elif arm in SEARCH_ARMS:
        store = GitTurns(workdir)

        def standing() -> str:
            return f"{SEARCH_ARMS[arm]}\n{found[0]}" if found[0] else ""
    elif arm in MEM0_ARMS:
        from studies.long_run.mem0_store import Mem0Turns

        store = Mem0Turns(mem0())

        def standing() -> str:
            return f"{MEM0_ARMS[arm]}\n{found[0]}" if found[0] else ""
    elif arm == "taste_rules":
        read, reader_calls = make_reader(chat, cut_off=reader_cut)
        store = GitTurns(workdir, header=lambda number, turn: CommitHeader(instructions=tuple(read(turn["content"])))
                         if turn["role"] == "user" and not turn.get("from_harness") else CommitHeader())

        def standing() -> str:
            kept = keep_newest([f"- {text}" for text in dict.fromkeys(store.instructions())], MEMORY_BUDGET, count)
            return f"{DELIVERED_HEADER}\n" + "\n".join(kept) if kept else ""
    elif arm in NOTE_ARMS:
        store = NotesStore(workdir, [], history=True)
        on_compact = make_note_taker(chat, NOTE_ARMS[arm], store, notes_skipped)

        def standing() -> str:
            newest = store.newest()
            return f"{NOTES_HEADER}\n{fill([newest], MEMORY_BUDGET, count)}" if newest else ""
    elif arm == "restated":

        def standing() -> str:
            said = [rule.get("restate") or rule["text"] for rule in rules
                    if rule["say"] == "first" or agent.n_calls + 1 >= rule["say"]]
            return f"{DELIVERED_HEADER}\n" + "\n".join(f"- {text}" for text in said) if said else ""
    actions = out / "actions.jsonl"
    if actions.exists():
        raise FileExistsError(f"{actions} already holds a run: a session has no resume, so give it a folder of its own")
    prompts = out / "prompts.jsonl"

    def on_prompt(row: dict) -> None:
        with prompts.open("a") as written:
            written.write(json.dumps(row) + "\n")

    def on_action(row: dict) -> None:
        with actions.open("a") as written:
            written.write(json.dumps(row) + "\n")

    summary_replies = {"asked": 0, "not_a_summary": 0}

    def on_summary_reply(row: dict) -> None:
        summary_replies["asked"] += 1
        summary_replies["not_a_summary"] += not row["summary"]
        with (out / "summary_replies.jsonl").open("a") as written:
            written.write(json.dumps({"call": agent.n_calls, **row}) + "\n")

    summarize = None if arm == "no_compaction" else make_summarizer(chat, session.get("summary_prompt", "anthropic"), package_prompt, on_reply=on_summary_reply)
    agent = LongRunAgent(
        ToolModel(chat, memory_tools=arm in TOOL_ARMS, observation_template=session["observation_template"],
                  format_error_template=session["format_error_template"]),
        MemoryEnvironment(None, store.answer if arm in TOOL_ARMS else None),
        count=count, summarize=summarize, record=store.add if isinstance(store, GitTurns) or arm in MEM0_ARMS else None, on_action=on_action,
        standing=standing, on_compact=on_compact, on_prompt=on_prompt,
        system_template=session["system_template"] + _SAID_ABOUT_THE_RECORD.get(arm, ""),
        instance_template=session["instance_template"], step_limit=session["step_limit"], cost_limit=0,
        compact_at_tokens=0 if arm == "no_compaction" else session.get("compact_at_tokens", 0),
        keep_recent_tokens=session.get("keep_recent_tokens", 0), task_calls=session["task_calls"],
        stop_at_tokens=session["stop_at_tokens"],
        planted={rule["say"]: rule["text"] for rule in rules if isinstance(rule["say"], int)},
        watch={rule["name"]: rule.get("watch") or rule["text"] for rule in rules},
        mentions={rule["name"]: rule["mentions"] for rule in rules if rule.get("mentions")})
    agent.extra_template_vars |= platform.uname()._asdict()  # mini-swe-agent's own templates ask for these

    def where_it_is(rule: dict, machines: int) -> dict[str, bool]:
        """Whether a rule had been told before this search, and where it stands now: in what the search brought, and in the conversation. A fact watched by its value is back
        only with that value; whether the words asking for it came too is read apart."""
        watched = (rule.get("watch") or rule["text"]).casefold()
        told = machines > 0 if rule["say"] == "first" else agent.n_calls >= rule["say"]
        talk = [render(message) for message in agent.messages[1:]]
        return {"told_before": told, "in_block": watched in found[0].casefold(), "instruction_in_block": rule["text"].casefold() in found[0].casefold(),
                "in_talk": any(watched in text.casefold() or (rule.get("mentions") and re.search(rule["mentions"], text)) for text in talk)}

    def look_up(task: dict, machines: int) -> None:
        """The harness's own look, as a new task comes in and before it goes into the store: one search with the task's own words, every call and what came back on the record.
        Before anything is stored there is nothing to search."""
        hits, tokens, calls = [], 0, []
        found[0] = ""
        if arm in MEM0_ARMS:
            store.flush()
        if store.commits:
            request = task.get("own_words") or task["instruction"]
            found[0], calls, hits, tokens = (store.search_with(request, MEMORY_BUDGET, count) if arm in MEM0_ARMS
                                             else search_with(store, request, MEMORY_BUDGET, count))
        searches.append({"task": task["name"], "call": agent.n_calls, "compactions": len(agent.events), "hits": len(hits), "tokens": tokens,
                         "rules": {rule["name"]: where_it_is(rule, machines) for rule in rules}})
        with (out / "searches.jsonl").open("a") as written:
            written.write(json.dumps({"task": task["name"], "call": agent.n_calls, "block": found[0], "calls": calls}) + "\n")

    def worded(task: dict, machines: int) -> str:
        """The rules told with the first task to get a machine, and never again in a message."""
        if arm in SEARCH_ARMS or arm in MEM0_ARMS:
            look_up(task, machines)
        told = [rule["text"] for rule in rules if rule["say"] == "first" and machines == 0]
        return "\n\n".join([*told, task["instruction"]])

    planted_on: list[str] = []
    walls: list[dict] = []

    def must(env: Any, command: str) -> None:
        done = env.execute({"command": command})
        if done.get("returncode"):
            raise RuntimeError(f"the harness's own step failed: {str(done.get('output'))[-300:]}")

    def prepared(task: dict) -> Any:
        """A task's machine, with anything the session plants there first: the harness's own step, in no message and no log. What is
        planted once goes on the first machine that actually takes it; one that will not take it leaves its task, like a machine that will not build."""
        env = make_env(task)
        try:
            for command in task.get("prepare", ()):
                must(env, command)
            if not planted_on:
                for command in session.get("first_prepare", ()):
                    must(env, command)
                planted_on.append(task["name"])
        except BaseException:
            getattr(env, "cleanup", lambda: None)()
            raise
        walls.append({"task": task["name"], "wall": getattr(env, "wall", None)})
        return env

    ended_by = None
    mem0_stats = None
    try:
        try:
            agent.run_tasks(session["tasks"], prepared, worded=worded)
        except (Exception, Stopped) as error:  # a failed model call or machine, or the job being stopped, ends the session, not its record
            ended_by = f"{type(error).__name__}: {error}"[:1000]
        disarm()
        mode = "agent_tools" if arm in TOOL_ARMS else "single_query" if arm in SEARCH_ARMS and store.store.memory_counts["search"] else "delivered"
        ledger = store.ledger(mode) if store else stand_in_record(BASELINE, code=f"studies/long_run/session.py:{arm}")
        note_versions = store.commits if arm in NOTE_ARMS else 0
        ledger_refused = None
        try:
            check([ledger])
        except ClaimRefused as error:
            ledger_refused = str(error)
    finally:
        if store and arm in MEM0_ARMS:
            try:  # what was still waiting goes in, and Mem0's own counts are read, before its process stops. A session that
                store.flush()  # ended early does not wait for Mem0's last adds: its record says how many were still waiting
                if ended_by is None:
                    store.drain()
                mem0_stats = store.stats()
            except Exception as error:
                mem0_stats = {"error": f"{type(error).__name__}: {error}"[:1000]}
        if store:
            store.close()
    rows = [json.loads(line) for line in actions.read_text().splitlines()] if actions.exists() else []
    silent = [row for row in rows if row["kind"] == "no_command"]
    reasons: dict[str, int] = {}
    for row in silent:
        reasons[str(row.get("finish_reason"))] = reasons.get(str(row.get("finish_reason")), 0) + 1
    summary = {"arm": arm, "versions": versions or {}, "outcomes": agent.outcomes, "ended_by": ended_by, "calls": agent.n_calls,
               "compactions": len(agent.events), "compactions_folding_little": sum(event["tokens_before"] - event["tokens_after"] < THIN_FOLD for event in agent.events),
               "events": agent.events, "summary_replies": summary_replies, "folds_waited": agent.folds_waited, "memory_turns": agent.recorded, "ledger": ledger,
               "resent": agent.resent, "ledger_refused": ledger_refused, "planted_on": planted_on[0] if planted_on else None, "walls": walls,
               "machines_not_stopped": agent.not_stopped, "model_use": model_use, "notes_not_written": notes_skipped[0],
               "valid": not agent.resent and ledger_refused is None and (bool(planted_on) or not session.get("first_prepare")),
               "no_command": len(silent), "no_command_finish_reasons": reasons,
               "reader_calls": reader_calls[0], "reader_cut_off": reader_cut[0], "note_versions": note_versions, "searches": searches,
               "extractor": "the study's reader, studies/context_rot/criteria.py" if arm == "taste_rules" else None, "mem0": mem0_stats,
               "scores": score_session(rows, rules)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def package_prompt() -> Callable:
    """The benchmark package's two published summary prompts."""
    from conpact.prompts import summarization_prompt

    return summarization_prompt


def _environment(kind: str, timeout: int) -> Callable[[dict], Any]:
    """The task's own container, or the machine's shell for a dry run. The memory tools never reach any of these."""

    def make(task: dict) -> Any:
        settings = {"cwd": task.get("cwd", ""), "timeout": timeout, "env": {"PAGER": "cat", "MANPAGER": "cat"}}
        if kind == "local":
            from minisweagent.environments.local import LocalEnvironment

            return LocalEnvironment(**settings)
        if kind == "machine":
            from studies.long_run.machine import Machine

            return Machine(image=task["image"], **settings)
        if kind == "docker":
            from minisweagent.environments.docker import DockerEnvironment

            return DockerEnvironment(image=task["image"], **settings)
        raise ValueError(f"unknown environment {kind!r}")

    return make


def _stop_on_term() -> Callable[[], None]:
    """Until the returned function is called, the job's stop signal ends the session as a failure does, so that its record is still written. After that, or once it has
    come, it changes nothing: the record is being written, and the job's folder goes only when this process has ended."""
    armed = [True]

    def stop(number: int, frame: Any) -> None:
        if armed[0]:
            armed[0] = False
            raise Stopped(f"signal {number}")

    signal.signal(signal.SIGTERM, stop)

    def disarm() -> None:
        armed[0] = False

    return disarm


def _counter(tokenizer: str) -> Callable[[str], int]:
    """The reader's own tokenizer; "words" is for dry runs only."""
    if tokenizer == "words":
        return lambda text: len(text.split())
    from tokenizers import Tokenizer

    path = Path(tokenizer)
    loaded = Tokenizer.from_file(str(path / "tokenizer.json" if path.is_dir() else path))
    return lambda text: len(loaded.encode(text, add_special_tokens=False).ids)


def main(argv: list[str] | None = None) -> int:
    from studies.memory_probe.tool_loop import http_chat

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("session", "url", "model", "tokenizer", "out", "workdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--env", default="docker", choices=("local", "machine", "docker"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--writing-tokens", type=int, default=8192, help="the cap on a summary, a notes version or a rule reading: reasoning counts toward a cap, and a cut-off one is useless")
    parser.add_argument("--command-timeout", type=int, default=300)
    parser.add_argument("--mem0-command", help="for the Mem0 arm: the command that runs Python in Mem0's own environment, as one string")
    parser.add_argument("--mem0-embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--mem0-dims", type=int, default=384)
    arguments = parser.parse_args(argv)
    if arguments.arm in MEM0_ARMS and not arguments.mem0_command:
        parser.error(f"--arm {arguments.arm} needs --mem0-command")

    import minisweagent

    acting = http_chat(arguments.url, arguments.model, max_tokens=arguments.max_tokens, effort=EFFORT)
    writing = http_chat(arguments.url, arguments.model, max_tokens=arguments.writing_tokens, effort=EFFORT)
    session = load_session(json.loads(Path(arguments.session).read_text()))
    disarm = _stop_on_term()

    def mem0() -> Any:
        import shlex

        from studies.long_run.mem0_store import start_sidecar

        return start_sidecar(shlex.split(arguments.mem0_command), url=arguments.url, model=arguments.model, max_tokens=MEM0_MAX_TOKENS,
                             effort=EFFORT, thinking=MEM0_THINKING, thinking_max_tokens=MEM0_THINKING_MAX_TOKENS, embedder=arguments.mem0_embedder, dims=arguments.mem0_dims, log=Path(arguments.out) / "mem0.log",
                             workdir=Path(arguments.workdir))

    summary = run_session(session, arm=arguments.arm, chat=lambda messages, tools: acting(messages, tools) if tools else writing(messages, None),
                          make_env=_environment(arguments.env, arguments.command_timeout), count=_counter(arguments.tokenizer),
                          out=Path(arguments.out), workdir=Path(arguments.workdir), package_prompt=package_prompt(), disarm=disarm,
                          mem0=mem0 if arguments.arm in MEM0_ARMS else None,
                          versions={"mini_swe_agent": minisweagent.__version__, "model": arguments.model, "effort": EFFORT,
                                    **({"mem0_thinking": MEM0_THINKING, "mem0_max_tokens": MEM0_MAX_TOKENS,
                                        "mem0_thinking_max_tokens": MEM0_THINKING_MAX_TOKENS} if arguments.arm in MEM0_ARMS else {}), "env": arguments.env,
                                    "commit": os.environ.get("TASTE_COMMIT", ""), "summary_prompt": session.get("summary_prompt", "")})
    print(json.dumps({key: summary[key] for key in ("arm", "outcomes", "calls", "compactions", "memory_turns", "scores")}))
    if summary["ended_by"] or not summary["valid"]:
        print(f"this session did not end cleanly: ended by {summary['ended_by']}, valid {summary['valid']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
