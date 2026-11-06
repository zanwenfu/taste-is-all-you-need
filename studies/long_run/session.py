"""One session of the long run: a chain of real tasks in one conversation, with compaction on.

A session file names the tasks and the rules to plant. A rule is told with the first task or as a user message at a
chosen model call, and never again, except in the ceiling arm that restates it. Each rule carries the patterns that
score it from the command record.

    compaction     the summary alone: a baseline
    taste          the same, and every message committed to Taste's memory, with search and fetch offered beside bash
    taste_exists   the same, and the system prompt says a searchable record exists
    taste_check    the same, and it also says to check that record before answering a request
    taste_rules    compaction, each user message read once as it arrives, what it asks to keep doing written in that commit's
                   header, and all of that read back from the store and put in the system prompt at every call; no tools
    notes_told     compaction, and a notes file a model rewrites at each compaction from what is about to be folded away, told to keep
                   what the user asked it to keep doing; every version in Taste's memory, the newest in the system prompt at every call
    notes_neutral  the same, not told that
    no_compaction  a ceiling: the whole conversation, as long as it fits
    restated       a ceiling: compaction, with every rule told so far said again at each task

Names, per the study's rule: "taste" is "Taste memory, study-driven"; the others have no Taste code in the measured path.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studies.context_rot.repeated import round_prompt
from studies.long_run.agent import LongRunAgent, MemoryEnvironment, ToolModel
from studies.long_run.score import score_session
from studies.memory_bench.core import fill
from studies.memory_ledger import BASELINE, stand_in_record
from studies.memory_probe.cases import DELIVERED_HEADER, FLAT_HEADER, LOOK, SENTENCE
from studies.memory_probe.stores import GitTurns, NotesStore
from taste.memstore import CommitHeader

TOOL_ARMS = ("taste", "taste_exists", "taste_check")
NOTE_ARMS = {"notes_told": "told", "notes_neutral": "neutral"}
ARMS = ("compaction", *TOOL_ARMS, "taste_rules", *NOTE_ARMS, "no_compaction", "restated")
MEMORY_BUDGET = 4096
_SAID_ABOUT_THE_RECORD = {"taste_exists": f" {SENTENCE}", "taste_check": f" {SENTENCE} {LOOK}"}
# Duke's Apptainer has no user-id map, and its fakeroot helper needs a newer C library than the task images have. Without the
# helper the agent is still root inside and its writes stay; a package whose install changes file owners may only half install.
CONTAINER_ARGS = ["--contain", "--cleanenv", "--fakeroot", "--ignore-fakeroot-command"]
# A warning from the container runtime would come first in a command's output, where the agent looks for its submission.
CONTAINER_GLOBAL_ARGS = ["--silent"]
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


def make_summarizer(chat: Callable, option: str, package_prompt: Callable, *, tries: int = 3) -> Callable:
    """The published summary prompts, each later round given the previous summary. An empty summary is asked again."""

    def summarize(older: list[dict], previous: str | None) -> str:
        messages = round_prompt(option, previous, older, package_prompt)
        for _ in range(tries):
            choice = chat(messages, None).choices[0]
            text = (choice.message.content or "").strip()
            if choice.finish_reason == "length":
                raise RuntimeError("a summary was cut off by the output cap")
            if text:
                return text
        raise RuntimeError(f"the model gave no summary in {tries} tries")

    return summarize


def make_reader(chat: Callable) -> tuple[Callable[[str], list[str]], list[int]]:
    """The study's reader, asked once per distinct user message, as it arrives."""
    from studies.context_rot.criteria import parse_reply, reader_prompt

    known: dict[str, list[str]] = {}
    calls = [0]

    def read(text: str) -> list[str]:
        if text not in known:
            reply = chat(reader_prompt(text), None)
            calls[0] += 1
            if reply.choices[0].finish_reason == "length":
                raise RuntimeError("a rule reading was cut off by the output cap")
            known[text] = parse_reply(reply.choices[0].message.content or "", text)[0]
        return known[text]

    return read, calls


def make_note_taker(chat: Callable, style: str, store: NotesStore) -> Callable[[list[dict]], None]:
    """The study's note-taker: the whole file rewritten from what a compaction is about to fold away."""
    from studies.memory_probe.notes import _PROMPT, MAX_TURN_CHARS, STYLES

    def keep(older: list[dict]) -> None:
        part = "\n".join(f"[{turn['role']}]: {str(turn['content'])[:MAX_TURN_CHARS]}" for turn in older)
        wanted = _PROMPT.format(told=STYLES[style], notes=store.newest() or "(no notes yet)", part=part)
        reply = chat([{"role": "user", "content": wanted}], None)
        if reply.choices[0].finish_reason == "length":
            raise RuntimeError("a notes version was cut off by the output cap")
        store.add_version(reply.choices[0].message.content or "")

    return keep


def run_session(session: dict, *, arm: str, chat: Callable, make_env: Callable[[dict], Any], count: Callable[[str], int],
                out: Path, workdir: Path, package_prompt: Callable, versions: dict | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    out, workdir = Path(out), Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    rules = session["rules"]
    store: Any = None
    standing, on_compact, reader_calls = None, None, [0]
    if arm in TOOL_ARMS:
        store = GitTurns(workdir)
    elif arm == "taste_rules":
        read, reader_calls = make_reader(chat)
        store = GitTurns(workdir, header=lambda number, turn: CommitHeader(instructions=tuple(read(turn["content"])))
                         if turn["role"] == "user" else CommitHeader())

        def standing() -> str:
            found = store.instructions()
            return f"{DELIVERED_HEADER}\n" + fill([f"- {text}\n" for text in found], MEMORY_BUDGET, count) if found else ""
    elif arm in NOTE_ARMS:
        store = NotesStore(workdir, [], history=True)
        on_compact = make_note_taker(chat, NOTE_ARMS[arm], store)

        def standing() -> str:
            newest = store.newest()
            return f"{FLAT_HEADER}\n{fill([newest], MEMORY_BUDGET, count)}" if newest else ""
    actions = out / "actions.jsonl"

    def on_action(row: dict) -> None:
        with actions.open("a") as written:
            written.write(json.dumps(row) + "\n")

    summarize = None if arm == "no_compaction" else make_summarizer(chat, session.get("summary_prompt", "anthropic"), package_prompt)
    agent = LongRunAgent(
        ToolModel(chat, memory_tools=arm in TOOL_ARMS, observation_template=session["observation_template"],
                  format_error_template=session["format_error_template"]),
        MemoryEnvironment(None, store.answer if arm in TOOL_ARMS else None),
        count=count, summarize=summarize, record=store.add if isinstance(store, GitTurns) else None, on_action=on_action,
        standing=standing, on_compact=on_compact,
        system_template=session["system_template"] + _SAID_ABOUT_THE_RECORD.get(arm, ""),
        instance_template=session["instance_template"], step_limit=session["step_limit"], cost_limit=0,
        compact_at_tokens=0 if arm == "no_compaction" else session.get("compact_at_tokens", 0),
        keep_recent_tokens=session.get("keep_recent_tokens", 0), task_calls=session["task_calls"],
        stop_at_tokens=session["stop_at_tokens"],
        planted={rule["say"]: rule["text"] for rule in rules if isinstance(rule["say"], int)},
        watch={rule["name"]: rule.get("watch") or rule["text"] for rule in rules})
    agent.extra_template_vars |= platform.uname()._asdict()  # mini-swe-agent's own templates ask for these

    def worded(task: dict, number: int) -> str:
        says = (lambda rule: rule.get("restate") or rule["text"]) if arm == "restated" else (lambda rule: rule["text"])
        told = [says(rule) for rule in rules if rule["say"] == "first" and (number == 0 or arm == "restated")]
        if arm == "restated":
            told += [says(rule) for rule in rules if isinstance(rule["say"], int) and agent.n_calls >= rule["say"]]
        return "\n\n".join([*told, task["instruction"]])

    def prepared(task: dict) -> Any:
        """A task's machine, with anything the session plants there first: the harness's own step, in no message and no log."""
        env = make_env(task)
        for command in task.get("prepare", ()):
            env.execute({"command": command})
        return env

    ended_by = None
    try:
        try:
            agent.run_tasks(session["tasks"], prepared, worded=worded)
        except Exception as error:  # a failed model call or machine ends the session, not its record
            ended_by = f"{type(error).__name__}: {error}"[:1000]
        ledger = (store.ledger("agent_tools" if arm in TOOL_ARMS else "delivered") if store
                  else stand_in_record(BASELINE, code=f"studies/long_run/session.py:{arm}"))
        note_versions = store.commits if arm in NOTE_ARMS else 0
    finally:
        if store:
            store.close()
    rows = [json.loads(line) for line in actions.read_text().splitlines()] if actions.exists() else []
    summary = {"arm": arm, "versions": versions or {}, "outcomes": agent.outcomes, "ended_by": ended_by, "calls": agent.n_calls,
               "compactions": len(agent.events), "events": agent.events, "memory_turns": agent.recorded, "ledger": ledger,
               "reader_calls": reader_calls[0], "note_versions": note_versions,
               "extractor": "the study's reader, studies/context_rot/criteria.py" if arm == "taste_rules" else None,
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
        if kind == "singularity":
            from minisweagent.environments.singularity import SingularityEnvironment

            return SingularityEnvironment(image=task["image"], executable="apptainer", exec_args=CONTAINER_ARGS,
                                          global_args=CONTAINER_GLOBAL_ARGS, **settings)
        if kind == "docker":
            from minisweagent.environments.docker import DockerEnvironment

            return DockerEnvironment(image=task["image"], **settings)
        raise ValueError(f"unknown environment {kind!r}")

    return make


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
    parser.add_argument("--env", default="docker", choices=("local", "singularity", "docker"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--command-timeout", type=int, default=300)
    arguments = parser.parse_args(argv)

    import minisweagent

    summary = run_session(load_session(json.loads(Path(arguments.session).read_text())), arm=arguments.arm,
                          chat=http_chat(arguments.url, arguments.model, max_tokens=arguments.max_tokens),
                          make_env=_environment(arguments.env, arguments.command_timeout), count=_counter(arguments.tokenizer),
                          out=Path(arguments.out), workdir=Path(arguments.workdir), package_prompt=package_prompt(),
                          versions={"mini_swe_agent": minisweagent.__version__, "model": arguments.model, "env": arguments.env})
    print(json.dumps({key: summary[key] for key in ("arm", "outcomes", "calls", "compactions", "memory_turns", "scores")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
