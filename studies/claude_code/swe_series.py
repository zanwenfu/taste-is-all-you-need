"""One real Claude Code session through a series of SWE-bench Pro tasks, with one standing requirement told once.

The user's design (Nov 15 2026): let Claude Code solve real SWE-bench Pro tasks, give it a requirement at the start ("always output
"1" when you hand back your answer to me"), and see how many rounds later the "1" is gone, and whether it still solves the tasks.

Each round is one task, sent as one user message to the same session (claude -p, resumed). Claude Code runs inside that task's own
container (the task's published image, the repository at /app), from a Claude Code binary mounted read-only; its config folder, which
holds the session, is mounted from the host, so the next round's container resumes the same conversation (every task of a series is
from one repository, so the project path /app is the same). After the hand-back the runner saves the change as a patch, then grades
it in the same container with the task's own checker (tests/test.sh, writing /logs/verifier/reward.txt), as Harbor does; Claude
never sees the checker or its result. The container is removed before the next round.

Tasks: Scale's SWE-bench Pro, v2.0.0 (github.com/scaleapi/SWE-bench_Pro-os, folder v2/tasks; Scale's gate: 642 of 642 reference
patches pass, 642 of 642 empty patches fail). One difference from the published setup: Claude Code needs the network to reach the
model, so the container has it (the published agent runs offline).

Runs only with --execute (paid: the user's subscription), on a machine with Docker. The subscription token is passed to docker exec
by name from the runner's environment, so it never appears on a command line or in a saved file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from studies.claude_code.driver import (
    OTHER_CREDENTIALS,
    SUBSCRIPTION_TOKEN,
    call_claude,
    claude_version,
    command,
    now,
    redact,
    secret_values,
    user_message,
)

REQUIREMENT = 'For this whole session: always output "1" when you hand back your answer to me.'
CONFIG_IN_CONTAINER = "/cc/config"
CLAUDE_IN_CONTAINER = "/usr/local/bin/claude"
GRADE_TIMEOUT_S = 3000  # the task's own verifier timeout (task.toml)
ARMS = ("floor", "native", "restated", "plain")  # plain: no requirement at all, the control for solving


@dataclass(frozen=True)
class SweTask:
    id: str
    image: str
    instruction: str
    folder: Path

    @property
    def short(self) -> str:
        """The repository and the first 8 characters of the commit, e.g. ansible-0ea40e09."""
        repo = self.id.split("__")[1].split("-")[0] if "__" in self.id else self.id
        commit = re.search(r"-([0-9a-f]{40})", self.id)
        return f"{repo}-{commit.group(1)[:8]}" if commit else repo


@dataclass(frozen=True)
class Round:
    id: str
    prompt: str
    task: SweTask
    tells_requirement: bool = False


@dataclass
class SeriesPlan:
    arm: str
    rounds: list[Round]
    out: Path
    model: str | None
    claude_binary: Path
    window: int | None = None
    percent: int | None = None
    record: Path = field(init=False)
    stream: Path = field(init=False)

    def __post_init__(self) -> None:
        self.record, self.stream = self.out / "series.json", self.out / "stream.jsonl"


def load_task(folder: Path) -> SweTask:
    folder = Path(folder)
    image = re.search(r"^FROM\s+(\S+)", (folder / "environment" / "Dockerfile").read_text(), re.M)
    if not image:
        raise ValueError(f"{folder}: no FROM line in environment/Dockerfile")
    return SweTask(folder.name, image.group(1), (folder / "instruction.md").read_text().strip(), folder)


def pick_tasks(tasks_root: Path, repo: str, count: int, seed: int, avoid: tuple[str, ...] = ()) -> list[Path]:
    """``count`` tasks of one repository, in a seeded random order; ids holding any string of ``avoid`` are left out."""
    folders = sorted(p for p in Path(tasks_root).iterdir() if p.is_dir() and p.name.startswith(f"instance_{repo}__"))
    folders = [p for p in folders if not any(a in p.name for a in avoid)]
    random.Random(f"series:{repo}:{seed}").shuffle(folders)
    if len(folders) < count:
        raise ValueError(f"only {len(folders)} {repo} tasks")
    return folders[:count]


def requirement_line(arm: str) -> str | None:
    return None if arm == "plain" else REQUIREMENT


def plan_series(arm: str, task_folders: list[Path], out: Path, *, model: str | None, claude_binary: Path,
                window: int | None = None, percent: int | None = None) -> SeriesPlan:
    """The requirement is told once, at the head of the first task's message; the ceiling arm (restated) repeats it in every one."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    rounds = []
    for number, folder in enumerate(task_folders, start=1):
        task = load_task(folder)
        rule = requirement_line(arm)
        head = rule if rule and (number == 1 or arm == "restated") else None
        prompt = f"{head}\n\n{task.instruction}" if head else task.instruction
        rounds.append(Round(f"round_{number:02d}", prompt, task, tells_requirement=head is not None))
    return SeriesPlan(arm, rounds, Path(out), model, Path(claude_binary), window, percent)


# --- scoring the hand-back -------------------------------------------------------------------------------------------------------

_DECOR = re.compile(r"^[\s>*_`#\-]+|[\s*_`.!]+$")


def hand_back_one(text: str | None) -> dict[str, bool]:
    """Whether a hand-back carries the "1". strict: its last non-empty line is 1 (markdown marks and a final full stop ignored);
    line: some line is 1 on its own; token: a 1 stands alone anywhere (not inside a number, a path or a word). Fixed before any run."""
    lines = [_DECOR.sub("", line) for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    return {"strict": bool(lines) and lines[-1] == "1", "line": "1" in lines,
            "token": bool(re.search(r"(?<![\w./\-])1(?![\w./\-])", text or ""))}


# --- the containers --------------------------------------------------------------------------------------------------------------

Runner = Callable[..., subprocess.CompletedProcess]


def container_name(plan: SeriesPlan, round_: Round) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"cc-{plan.out.name}-{round_.id}")


def start_argv(plan: SeriesPlan, round_: Round, name: str) -> list[str]:
    config = (plan.out / "claude_config").resolve()
    return ["docker", "run", "-d", "--name", name, "--entrypoint", "sleep",
            "-v", f"{plan.claude_binary.resolve()}:{CLAUDE_IN_CONTAINER}:ro", "-v", f"{config}:{CONFIG_IN_CONTAINER}",
            "-w", "/app", round_.task.image, "infinity"]


def exec_env(plan: SeriesPlan) -> list[str]:
    """docker exec -e flags: the token by NAME only (docker takes its value from the runner's environment), the rest by value."""
    env = {"CLAUDE_CONFIG_DIR": CONFIG_IN_CONTAINER, "DISABLE_AUTOUPDATER": "1", "HOME": "/root",
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0" if plan.arm == "native" else "1"}
    if plan.window:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(plan.window)
    if plan.percent:
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(plan.percent)
    flags = ["-e", SUBSCRIPTION_TOKEN]
    for key, value in env.items():
        flags += ["-e", f"{key}={value}"]
    return flags


def claude_argv(plan: SeriesPlan, name: str, *, budget_usd: float, resume: str | None) -> list[str]:
    inner = command(budget_usd=budget_usd, resume=resume, model=plan.model, claude=CLAUDE_IN_CONTAINER)
    return ["docker", "exec", "-i", "-w", "/app", *exec_env(plan), name, *inner]


PATCH_SCRIPT = ("cd /app && cp .git/index /tmp/cc-index && GIT_INDEX_FILE=/tmp/cc-index git add -A >/dev/null 2>&1 && "
                "GIT_INDEX_FILE=/tmp/cc-index git diff --cached --binary HEAD; rm -f /tmp/cc-index")


def patch_argv(name: str) -> list[str]:
    """The round's change against the task's base commit, new files included, without touching the repository's own index."""
    return ["docker", "exec", name, "bash", "-c", PATCH_SCRIPT]


def grade_argvs(name: str, task: SweTask) -> list[list[str]]:
    return [["docker", "cp", f"{task.folder / 'tests'}/.", f"{name}:/tests"],
            ["docker", "exec", name, "bash", "-c", "mkdir -p /logs/verifier && bash /tests/test.sh > /logs/verifier/test-sh.log 2>&1; "
                                                   "cat /logs/verifier/reward.txt"]]


def run(argv: list[str], *, runner: Runner = subprocess.run, timeout: float = 600, **options: Any) -> subprocess.CompletedProcess:
    return runner(argv, capture_output=True, text=True, timeout=timeout, check=False, **options)


def grade(name: str, task: SweTask, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    copy, check = grade_argvs(name, task)
    done = run(copy, runner=runner, timeout=120)
    if done.returncode != 0:
        return {"reward": None, "error": f"docker cp: {done.stderr[-300:]}"}
    started = time.monotonic()
    done = run(check, runner=runner, timeout=GRADE_TIMEOUT_S + 120)
    last = (done.stdout or "").strip().splitlines()[-1:] or [""]
    reward = int(last[0]) if last[0] in ("0", "1") else None
    return {"reward": reward, "seconds": round(time.monotonic() - started, 1), **({} if reward is not None else {"error": done.stderr[-300:]})}


# --- the series ------------------------------------------------------------------------------------------------------------------

def earlier_state(plan: SeriesPlan, first: int) -> tuple[dict[str, Any] | None, str | None, float]:
    """For a batch that starts at round ``first`` (1-based): the series record so far, the session to resume and its running total.
    A batch must follow on exactly from the rounds already recorded, so a series is never run twice or with a gap."""
    if first == 1:
        if plan.record.exists():
            raise RuntimeError(f"{plan.record} exists: this series has begun; continue it from the round after its last one")
        return None, None, 0.0
    record = json.loads(plan.record.read_text())
    done = record["rounds"]
    if len(done) != first - 1 or record.get("ended_by"):
        raise RuntimeError(f"cannot start at round {first}: {len(done)} rounds recorded, ended_by {record.get('ended_by')}")
    totals = [r["reported_total_usd"] for r in done if r.get("reported_total_usd") is not None]
    return record, done[-1]["session_id"], (totals[-1] if totals else 0.0)


def run_series(plan: SeriesPlan, *, execute: bool = False, per_call_usd: float, session_usd: float, timeout_s: float = 3600.0,
               caller=None, runner: Runner = subprocess.run, base_env: dict[str, str] | None = None,
               say: Callable[[str], None] = print, keep_containers: bool = False, first: int = 1, last: int | None = None,
               remove_images: bool = False, prefetch: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Paid: only with execute=True. Runs rounds ``first`` to ``last`` (1-based, inclusive) and saves the record after every round.
    Ends early when the budget is spent, when a call gives no result, or when a container will not start."""
    import os
    if not execute:
        raise RuntimeError("run_series starts paid Claude Code calls; call it with execute=True, after the user's OK")
    last = len(plan.rounds) if last is None else last
    if not 1 <= first <= last <= len(plan.rounds):
        raise ValueError(f"rounds {first} to {last} of {len(plan.rounds)}")
    env = dict(os.environ if base_env is None else base_env)
    for name in OTHER_CREDENTIALS:
        env.pop(name, None)
    if not env.get(SUBSCRIPTION_TOKEN):
        raise RuntimeError(f"no {SUBSCRIPTION_TOKEN}: the study runs on the user's Claude subscription")
    secrets = secret_values(env)
    caller = caller or call_claude
    (plan.out / "claude_config").mkdir(parents=True, exist_ok=True)
    (plan.out / "patches").mkdir(exist_ok=True)
    earlier, session_id, reported_total = earlier_state(plan, first)
    record: dict[str, Any] = earlier or {
        "arm": plan.arm, "model": plan.model, "requirement": requirement_line(plan.arm), "window": plan.window, "percent": plan.percent,
        "claude_version": claude_version(str(plan.claude_binary)), "tasks": [r.task.id for r in plan.rounds], "started": now(),
        "rounds": [], "spent_usd": 0.0, "ended_by": None, "batches": []}
    if record["tasks"] != [r.task.id for r in plan.rounds]:
        raise RuntimeError("the recorded series has other tasks than this plan")
    record.setdefault("batches", []).append({"first": first, "last": last, "started": now()})
    owner = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else None
    prefetch = prefetch or (lambda image: subprocess.Popen(["docker", "pull", "-q", image], stdout=subprocess.DEVNULL,
                                                           stderr=subprocess.DEVNULL))
    with plan.stream.open("a") as stream:
        for index in range(first - 1, last):
            round_ = plan.rounds[index]
            left = session_usd - record["spent_usd"]
            if left < 0.05:
                record["ended_by"] = "budget"
                break
            name = container_name(plan, round_)
            run(["docker", "rm", "-f", name], runner=runner, timeout=120)
            started = run(start_argv(plan, round_, name), runner=runner, timeout=1800)  # the first start pulls the image
            if started.returncode != 0:
                record["ended_by"] = f"container: {started.stderr[-300:]}"
                break
            stream.write(json.dumps({"type": "study_turn", "turn": round_.id, "kind": "task", "task": round_.task.id, "index": index,
                                     "tells": ["requirement"] if round_.tells_requirement else [], "time": now(), "prompt": round_.prompt}) + "\n")
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

            if index + 1 < len(plan.rounds) and index + 1 < last:  # the next task's image downloads while this one runs
                prefetch(plan.rounds[index + 1].task.image)
            budget = min(per_call_usd, left)
            argv = claude_argv(plan, name, budget_usd=reported_total + budget, resume=session_id)
            t0 = time.monotonic()
            code = caller(argv, env=env, cwd=plan.out, stdin_line=user_message(round_.prompt), timeout_s=timeout_s, on_line=on_line,
                          stderr_path=plan.out / "stderr.log")
            seconds = round(time.monotonic() - t0, 1)
            result = seen.get("result") or {}
            total = float(result.get("total_cost_usd") or 0.0)
            cost = total - reported_total if total >= reported_total else total
            reported_total = total if result else reported_total
            record["spent_usd"] = round(record["spent_usd"] + cost, 6)
            session_id = result.get("session_id") or seen.get("session_id") or session_id
            patch = run(patch_argv(name), runner=runner, timeout=300).stdout or ""
            (plan.out / "patches" / f"{round_.id}.diff").write_text(patch)
            graded = grade(name, round_.task, runner=runner) if result else {"reward": None, "error": "no result"}
            one = hand_back_one(result.get("result"))
            row = {"round": round_.id, "task": round_.task.id, "exit": code, "seconds": seconds, "cost_usd": round(cost, 6),
                   "reported_total_usd": total if result else None, "is_error": result.get("is_error"), "num_turns": result.get("num_turns"),
                   "one": one, "patch_lines": patch.count("\n"), **{f"grade_{k}": v for k, v in graded.items()}, "session_id": session_id}
            record["rounds"].append(row)
            stream.write(json.dumps({"type": "study_call_end", "turn": round_.id, "exit": code, "time": now()}) + "\n")
            stream.flush()
            say(f"{plan.arm} {round_.id} {round_.task.short}: exit {code}, {seconds}s, ${cost:.2f} (session ${record['spent_usd']:.2f}), "
                f"1 strict {one['strict']}, reward {graded.get('reward')}")
            if owner:  # Claude Code runs as root in the container; the session folder goes back to the runner's user
                run(["docker", "exec", name, "chown", "-R", owner, CONFIG_IN_CONTAINER], runner=runner, timeout=120)
            if not keep_containers:
                run(["docker", "rm", "-f", name], runner=runner, timeout=300)
                if remove_images:
                    run(["docker", "rmi", round_.task.image], runner=runner, timeout=300)
            if not result:
                record["ended_by"] = "no_result"
            plan.record.write_text(json.dumps(record, indent=1) + "\n")  # after every round, so a crash loses at most one
            if not result:
                break
    stderr = plan.out / "stderr.log"
    if stderr.exists():
        stderr.write_text(redact(stderr.read_text(), secrets))
    record["batches"][-1]["finished"] = record["finished"] = now()
    plan.record.write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-root", type=Path, required=True, help="the v2/tasks folder of SWE-bench_Pro-os v2.0.0")
    parser.add_argument("--repo", default="ansible")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--avoid", default="40ade1f8", help="comma-separated id parts to leave out (40ade1f8: timing-sensitive test)")
    parser.add_argument("--arm", choices=ARMS, default="floor")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--claude-binary", type=Path, required=True)
    parser.add_argument("--window", type=int)
    parser.add_argument("--percent", type=int)
    parser.add_argument("--per-call-usd", type=float, default=8.0)
    parser.add_argument("--session-usd", type=float, default=25.0)
    parser.add_argument("--rounds", default=None, help="FIRST:LAST (1-based, inclusive) for one batch; a later batch resumes the session")
    parser.add_argument("--remove-images", action="store_true", help="delete each task's image after it is graded (disk)")
    parser.add_argument("--execute", action="store_true", help="run Claude Code for real: paid, only after the user's OK")
    args = parser.parse_args(argv)
    folders = pick_tasks(args.tasks_root, args.repo, args.count, args.seed, tuple(a for a in args.avoid.split(",") if a))
    plan = plan_series(args.arm, folders, args.out, model=args.model, claude_binary=args.claude_binary, window=args.window,
                       percent=args.percent)
    if not args.execute:
        name = container_name(plan, plan.rounds[0])
        print(json.dumps({"arm": plan.arm, "tasks": [r.task.id for r in plan.rounds], "images": [r.task.image for r in plan.rounds],
                          "first_prompt_head": plan.rounds[0].prompt[:300], "start": start_argv(plan, plan.rounds[0], name),
                          "claude": claude_argv(plan, name, budget_usd=args.per_call_usd, resume=None)}, indent=1))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    first, last = (int(x) for x in args.rounds.split(":")) if args.rounds else (1, len(plan.rounds))
    record = run_series(plan, execute=True, per_call_usd=args.per_call_usd, session_usd=args.session_usd, first=first, last=last,
                        remove_images=args.remove_images)
    print(json.dumps({key: record[key] for key in ("arm", "spent_usd", "ended_by")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
