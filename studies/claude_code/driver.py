"""One Claude Code session of the study: the turns it is sent, the exact command line and environment of each call, the project
folder it works in, and, only when asked in so many words, the run itself.

A session is a chain of small tasks (tasks.py) sent to headless Claude Code one turn at a time, each call resuming the session of the
call before it. Three standing rules are told in chat, as the long-run study tells them: Python's -B flag and the session tag in the
first task's message, and ls -1 in a message of its own before the fourth task. The project's CLAUDE.md, written here, says to list
folders with ls -la, so the chat's ls rule overrides the file for the session, and the two compete after every compaction.

The arms differ only in the environment, the project's hooks (settings.py) and, for the ceiling, the prompts:

    floor     auto memory off; the chat's rules live only in the conversation and its summaries
    native    Claude Code as it ships: auto memory on, its MEMORY.md given back after compaction
    rules     floor, and a reader-kept rule list returned with every prompt and after every compaction (hooks/rule_list_hook.py)
    mem0      floor, and Mem0 searched with every prompt and given every finished exchange (hooks/mem0_hook.py)
    restated  floor, and every rule told so far appended to every later task's prompt: a ceiling

Every call is the same command line in every arm: claude -p with stream-json in and out, --verbose, the six tools allowed, a permission
mode that runs them unattended, a cost cap, and --resume from the second call on. Never --bare, which would skip CLAUDE.md, auto
memory and hooks. CLAUDE_CONFIG_DIR points into the session's folder, so no user settings, user CLAUDE.md or earlier memory leak in,
and the session's transcripts and auto memory are kept with its results.

    python -m studies.claude_code.driver --arm floor --seed 1 --tasks 30 --out RUNS/s1/floor           # prints the plan; runs nothing
    python -m studies.claude_code.driver ... --prepare                                                 # also writes the folder
    python -m studies.claude_code.driver ... --execute --per-call-usd 2 --session-usd 15               # runs it: paid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studies.claude_code.credentials import OTHER_CREDENTIALS, SUBSCRIPTION_TOKEN  # noqa: F401  (re-exported)
from studies.claude_code.hooks.common import now
from studies.claude_code.hooks.mem0_hook import CLIENT_ENV
from studies.claude_code.hooks.rule_list_hook import READER_ENV
from studies.claude_code.settings import ARMS, write_settings
from studies.claude_code.tasks import Task, make_tasks, write_starting_files

TOOLS = ("Bash", "Edit", "Write", "Read", "Glob", "Grep")
PYTHON_B = "For the rest of this session, whenever you run Python, pass it the -B flag, as in python3 -B script.py."
LS_ONE = "From now on, when you list a folder, use ls -1 and no other form of ls."
CLAUDE_MD = "# Project notes\n\nSmall Python and shell tools, one folder per task.\n\nWhen you list a folder, use ls -la.\n"
TAG_FILE = ".session_tag"
LS_RULE_BEFORE_TASK = 4
WINDOW = 100_000  # CLAUDE_CODE_AUTO_COMPACT_WINDOW; the docs allow 100,000 to 1,000,000
COMPACT_PERCENT = 50  # CLAUDE_AUTOCOMPACT_PCT_OVERRIDE; it can only lower the trigger
PERMISSION_MODE = "acceptEdits"  # edits run unattended; the allowed tools cover Bash; denials are counted from each result line
MIN_CALL_USD = 0.25  # a call given less than this is not started: it could not finish a task
RESUME_PLACEHOLDER = "<session id from the previous call>"
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


def session_tag(seed: int) -> str:
    """The session's tag, as the long-run study makes it."""
    return hashlib.sha256(f"session-tag-{seed}".encode()).hexdigest()[:12]


def tag_rule(removed: bool = True) -> str:
    gone = " It is removed after this task, so read it now." if removed else " Read it now."
    return (f"This project has a file, {TAG_FILE}, holding this session's tag.{gone} For the rest of this session, in every task, run "
            f"echo TAG once before you finish, with the real tag in place of TAG.")


def restated(tag: str, ls_told: bool) -> str:
    """The ceiling's block: every rule told so far, the tag given outright, as the long-run study's ceiling restates it."""
    rules = [PYTHON_B, f"This session's tag is {tag}. In every task, run echo {tag} once before you finish.", *([LS_ONE] if ls_told else [])]
    return "Standing rules for this session:\n" + "\n".join(f"- {rule}" for rule in rules)


@dataclass(frozen=True)
class Turn:
    id: str  # task_01 ... or rule_ls
    kind: str  # "task" or "rule"
    prompt: str
    tells: tuple[str, ...] = ()  # the rules this message tells, by the scorer's names
    task: Task | None = None


def plan_turns(arm: str, tasks: list[Task], tag: str, *, ls_rule_before: int = LS_RULE_BEFORE_TASK, tag_removed: bool = True) -> list[Turn]:
    """The messages of a session, in order. The rules are told the same way in every arm; only the ceiling restates them."""
    turns: list[Turn] = []
    for task in tasks:
        if task.number == ls_rule_before:
            turns.append(Turn("rule_ls", "rule", LS_ONE, ("ls_one",)))
        if task.number == 1:
            turns.append(Turn("task_01", "task", f"{PYTHON_B} {tag_rule(tag_removed)}\n\n{task.prompt}", ("python_B", "session_tag"), task))
            continue
        prompt = task.prompt
        if arm == "restated":
            prompt = f"{prompt}\n\n{restated(tag, ls_told=task.number >= ls_rule_before)}"
        turns.append(Turn(f"task_{task.number:02d}", "task", prompt, (), task))
    return turns


def command(*, budget_usd: float, resume: str | None = None, model: str | None = None, permission_mode: str = PERMISSION_MODE,
            claude: str = "claude") -> list[str]:
    """One call's command line, the same in every arm. The message goes on stdin as one stream-json line."""
    argv = [claude, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--allowedTools", *TOOLS,
            "--permission-mode", permission_mode, "--max-budget-usd", f"{budget_usd:.2f}"]
    argv += ["--model", model] if model else []
    argv += ["--resume", resume] if resume else []
    if "--bare" in argv:  # it would skip CLAUDE.md, auto memory and hooks: everything the study measures
        raise AssertionError("never --bare")
    return argv


def environment(arm: str, *, config_dir: Path, window: int = WINDOW, percent: int = COMPACT_PERCENT, reader: str | None = None,
                memory: str | None = None) -> dict[str, str]:
    """What the study sets on top of the caller's environment; nothing here is a secret, so it is saved with the session."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; the arms are {', '.join(ARMS)}")
    if not 100_000 <= window <= 1_000_000 or not 1 <= percent <= 100:
        raise ValueError("the compaction window must be 100,000 to 1,000,000 tokens and the percent 1 to 100")
    env = {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(window), "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(percent), "DISABLE_AUTOUPDATER": "1",
           "CLAUDE_CONFIG_DIR": str(config_dir), "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0" if arm == "native" else "1"}
    if arm == "rules":
        if not reader:
            raise ValueError("the rules arm needs a reader, as module:function")
        env[READER_ENV] = reader
    if arm == "mem0":
        if not memory:
            raise ValueError("the mem0 arm needs a memory client maker, as module:function")
        env[CLIENT_ENV] = memory
    return env


@dataclass(frozen=True)
class SessionPlan:
    arm: str
    seed: int
    tag: str
    out: Path
    turns: tuple[Turn, ...]
    env: dict[str, str]
    model: str | None = None
    permission_mode: str = PERMISSION_MODE
    claude: str = "claude"
    python: str = sys.executable
    tag_removed: bool = True

    @property
    def project(self) -> Path:
        return self.out / "project"

    @property
    def hooks_dir(self) -> Path:  # hook logs and Mem0's waiting prompt, outside the project
        return self.out / "hooks"

    @property
    def stream(self) -> Path:
        return self.out / "stream.jsonl"

    @property
    def record(self) -> Path:
        return self.out / "session.json"


def plan_session(arm: str, out: Path, *, seed: int, tasks: int = 30, ls_rule_before: int = LS_RULE_BEFORE_TASK, model: str | None = None,
                 reader: str | None = None, memory: str | None = None, window: int = WINDOW, percent: int = COMPACT_PERCENT,
                 permission_mode: str = PERMISSION_MODE, claude: str = "claude", python: str = sys.executable,
                 tag_removed: bool = True) -> SessionPlan:
    """Everything about one session, decided before anything is written or run."""
    out = Path(out).absolute()
    tag = session_tag(seed)
    env = environment(arm, config_dir=out / "claude_config", window=window, percent=percent, reader=reader, memory=memory)
    turns = plan_turns(arm, make_tasks(tasks, seed), tag, ls_rule_before=ls_rule_before, tag_removed=tag_removed)
    return SessionPlan(arm, seed, tag, out, tuple(turns), env, model, permission_mode, claude, python, tag_removed)


def user_message(prompt: str) -> str:
    """One user message as a stream-json input line, in the shape the Agent SDK sends."""
    return json.dumps({"type": "user", "message": {"role": "user", "content": prompt}, "parent_tool_use_id": None, "session_id": "default"})


def dry_run(plan: SessionPlan, per_call_usd: float) -> dict[str, Any]:
    """What would be sent, call by call, without running or writing anything."""
    calls = [{"turn": turn.id, "kind": turn.kind, "tells": list(turn.tells),
              "argv": command(budget_usd=per_call_usd, resume=RESUME_PLACEHOLDER if number else None, model=plan.model,
                              permission_mode=plan.permission_mode, claude=plan.claude),
              "stdin": user_message(turn.prompt)} for number, turn in enumerate(plan.turns)]
    return {"arm": plan.arm, "seed": plan.seed, "tag": plan.tag, "cwd": str(plan.project), "env": plan.env, "calls": calls}


def check_isolated(project: Path) -> None:
    """Refuse a project under a folder Claude Code would read instructions from: it loads CLAUDE.md files from every parent folder."""
    for parent in Path(project).absolute().parents:
        for name in ("CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md", ".git"):
            if (parent / name).exists():
                raise ValueError(f"{parent / name} is above the project; Claude Code would read it. Put the session's folder elsewhere.")


def prepare(plan: SessionPlan, *, isolated: bool = True) -> None:
    """Write the session's folder: the project with its CLAUDE.md, tag file and hooks; the config and hook folders. Refuses a folder
    already in use, so no session starts on another's leftovers."""
    if plan.out.exists() and any(plan.out.iterdir()):
        raise FileExistsError(f"{plan.out} is not empty")
    if isolated:
        check_isolated(plan.project)
    for folder in (plan.project, plan.hooks_dir, Path(plan.env["CLAUDE_CONFIG_DIR"])):
        folder.mkdir(parents=True, exist_ok=True)
    (plan.project / "CLAUDE.md").write_text(CLAUDE_MD)
    (plan.project / TAG_FILE).write_text(plan.tag + "\n")
    write_settings(plan.project, plan.arm, state=plan.hooks_dir, python=plan.python)


def secret_values(env: Mapping[str, str]) -> list[str]:
    """Values of variables whose names look secret, longest first, so that a saved line never holds one."""
    return sorted({value for name, value in env.items() if _SECRET_NAME.search(name) and len(value) >= 8}, key=len, reverse=True)


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text


def hook_errors(folder: Path) -> int:
    """Lines with an error in the hooks' logs: a hook that failed means the arm is not what it says, so the session stops."""
    count = 0
    for log in Path(folder).glob("*.jsonl"):
        for line in log.read_text().splitlines():
            try:
                count += "error" in json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
    return count


Caller = Callable[..., int]


def call_claude(argv: list[str], *, env: Mapping[str, str], cwd: Path, stdin_line: str, timeout_s: float, on_line: Callable[[str], None],
                stderr_path: Path) -> int:
    """One real Claude Code call: send the message, pass each stdout line on as it comes, return the exit code. A call over its time
    limit gets SIGTERM, which drops the turn in progress."""
    with open(stderr_path, "a") as errors:
        process = subprocess.Popen(argv, cwd=cwd, env=dict(env), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True,
                                   bufsize=1)
        timer = threading.Timer(timeout_s, process.terminate)
        timer.start()
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(stdin_line + "\n")
            process.stdin.close()
            for line in process.stdout:
                on_line(line)
            return process.wait()
        finally:
            timer.cancel()


def claude_version(claude: str) -> str:
    try:
        return subprocess.run([claude, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unknown ({type(error).__name__})"


def run(plan: SessionPlan, *, execute: bool = False, per_call_usd: float, session_usd: float, timeout_s: float = 1800.0,
        caller: Caller | None = None, base_env: Mapping[str, str] | None = None, extra_env: Mapping[str, str] | None = None,
        say: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the session: paid, so only with execute=True. Each call's lines go to stream.jsonl as they come, after a study_turn line
    naming the turn; a call is given at most what is left of the session's budget; the session ends when the budget is spent, when a
    call gives no result line, or when a hook logs an error. ``extra_env`` is the provider's settings (for example Bedrock's); nothing
    from the environment is saved or printed, and any value that looks secret is cut out of every saved line."""
    if not execute:
        raise RuntimeError("run() starts paid Claude Code calls; call it with execute=True, after the user's OK")
    if not plan.project.exists():
        prepare(plan)
    caller = caller or call_claude
    env = {**(os.environ if base_env is None else base_env), **(extra_env or {}), **plan.env}
    # The user's decision (Nov 15): the study runs on their Claude subscription, never on an API key. Any other credential in the
    # environment would make Claude Code bill it instead, so each is dropped, and the session will not start without the token.
    for name in OTHER_CREDENTIALS:
        env.pop(name, None)
    if not env.get(SUBSCRIPTION_TOKEN):
        raise RuntimeError(f"no {SUBSCRIPTION_TOKEN}: the study runs on the user's Claude subscription; give the token to the subprocess "
                           "from its file, never an API key")
    secrets = secret_values(env)
    record: dict[str, Any] = {"arm": plan.arm, "seed": plan.seed, "tag": plan.tag, "model": plan.model, "env": plan.env,
                              "permission_mode": plan.permission_mode, "per_call_usd": per_call_usd, "session_usd": session_usd,
                              "turns": [{"turn": t.id, "kind": t.kind, "tells": list(t.tells), "prompt": t.prompt} for t in plan.turns],
                              "claude_version": claude_version(plan.claude) if caller is call_claude else "not run",
                              "started": now(), "calls": [], "spent_usd": 0.0, "ended_by": None}
    session_id: str | None = None
    # Claude Code's total_cost_usd (and its modelUsage tokens) is the resumed session's running total, not one call's (the pilot,
    # Nov 15: the tokens grew from 70K to 659K cached over 11 calls). So a call's cost is the change in that total, and the cap given
    # to Claude Code is on the running total too.
    reported_total = 0.0
    with plan.stream.open("a") as stream:
        for number, turn in enumerate(plan.turns):
            left = session_usd - record["spent_usd"]
            if left < MIN_CALL_USD:
                record["ended_by"] = "budget"
                break
            if turn.task:
                write_starting_files(plan.project, turn.task)
            budget = min(per_call_usd, left)
            argv = command(budget_usd=reported_total + budget, resume=session_id, model=plan.model, permission_mode=plan.permission_mode,
                           claude=plan.claude)
            stream.write(json.dumps({"type": "study_turn", "turn": turn.id, "kind": turn.kind, "tells": list(turn.tells), "index": number,
                                     "time": now(), "prompt": turn.prompt}) + "\n")
            stream.flush()
            seen: dict[str, Any] = {}

            def on_line(line: str) -> None:
                stream.write(redact(line.rstrip("\n"), secrets) + "\n")
                stream.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    return
                if isinstance(event, dict) and event.get("session_id"):
                    seen["session_id"] = event["session_id"]
                if isinstance(event, dict) and event.get("type") == "result":
                    seen["result"] = event

            started = time.monotonic()
            code = caller(argv, env=env, cwd=plan.project, stdin_line=user_message(turn.prompt), timeout_s=timeout_s, on_line=on_line,
                          stderr_path=plan.out / "stderr.log")
            result = seen.get("result") or {}
            total = float(result.get("total_cost_usd") or 0.0)
            cost = total - reported_total if total >= reported_total else total  # a total that went down started afresh
            reported_total = total
            record["spent_usd"] = round(record["spent_usd"] + cost, 6)
            session_id = result.get("session_id") or seen.get("session_id") or session_id
            record["calls"].append({"turn": turn.id, "exit": code, "seconds": round(time.monotonic() - started, 1), "budget_usd": budget,
                                    "cost_usd": round(cost, 6), "reported_total_usd": total, "subtype": result.get("subtype"),
                                    "session_id": session_id})
            stream.write(json.dumps({"type": "study_call_end", "turn": turn.id, "exit": code, "time": now()}) + "\n")
            stream.flush()
            say(f"{plan.arm} {turn.id}: exit {code}, ${cost:.2f}, ${record['spent_usd']:.2f} of ${session_usd:.2f} spent")
            if not result:
                record["ended_by"] = "no_result"
                break
            if turn.id == "task_01" and plan.tag_removed:
                (plan.project / TAG_FILE).unlink(missing_ok=True)
            if hook_errors(plan.hooks_dir):
                record["ended_by"] = "hook_error"
                break
    stderr = plan.out / "stderr.log"
    if stderr.exists():
        stderr.write_text(redact(stderr.read_text(), secrets))
    record["finished"] = now()
    plan.record.write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--ls-rule-before", type=int, default=LS_RULE_BEFORE_TASK)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--reader", help="the rules arm's reader, as module:function")
    parser.add_argument("--memory", help="the mem0 arm's client maker, as module:function")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--percent", type=int, default=COMPACT_PERCENT)
    parser.add_argument("--permission-mode", default=PERMISSION_MODE)
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--keep-tag-file", action="store_true", help="leave .session_tag in the project after the first task")
    parser.add_argument("--per-call-usd", type=float, default=2.0)
    parser.add_argument("--session-usd", type=float, default=15.0)
    parser.add_argument("--prepare", action="store_true", help="write the session's folder; run nothing")
    parser.add_argument("--execute", action="store_true", help="run Claude Code for real: paid, only after the user's OK")
    args = parser.parse_args(argv)
    plan = plan_session(args.arm, args.out, seed=args.seed, tasks=args.tasks, ls_rule_before=args.ls_rule_before, model=args.model,
                        reader=args.reader, memory=args.memory, window=args.window, percent=args.percent,
                        permission_mode=args.permission_mode, claude=args.claude, tag_removed=not args.keep_tag_file)
    if args.execute:
        record = run(plan, execute=True, per_call_usd=args.per_call_usd, session_usd=args.session_usd)
        print(json.dumps({key: record[key] for key in ("arm", "seed", "spent_usd", "ended_by")}))
    elif args.prepare:
        prepare(plan)
        print(f"prepared {plan.out}")
    else:
        print(json.dumps(dry_run(plan, args.per_call_usd), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
