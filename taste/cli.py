"""``taste`` CLI — ``taste run <task> --agent <spec>``."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from taste.agent import AgentSpec
from taste.kernel import Event, Kernel, RunResult
from taste.llm import LLM
from taste.memory import Memory

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Agent OS — a git-native harness for long-running agents."""


@main.command("run")
@click.argument("task", nargs=-1, required=True)
@click.option(
    "--agent",
    "agent_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the agent's agent_desp.md spec file.",
)
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Git repo the agent is allowed to modify. Defaults to cwd.",
)
@click.option("--session", default=None, help="Session id (auto-generated if omitted).")
@click.option("--base-ref", default="HEAD", help="Git ref to branch from.")
@click.option("--max-retries", default=2, show_default=True, type=int)
def run_cmd(
    task: tuple[str, ...],
    agent_path: Path,
    workspace: Path,
    session: str | None,
    base_ref: str,
    max_retries: int,
) -> None:
    """Run an agent on TASK. Every step becomes a commit on a fresh branch."""
    spec = AgentSpec.from_file(agent_path)
    task_text = " ".join(task)

    console.print(
        Panel.fit(
            f"[bold]{spec.name}[/] — {spec.description}\n"
            f"[dim]workspace:[/] {workspace}\n"
            f"[dim]task:[/] {task_text}",
            title="taste run",
            border_style="cyan",
        )
    )

    llm = LLM()
    kernel = Kernel(
        workspace=workspace,
        llm=llm,
        max_retries=max_retries,
        on_event=_print_event,
    )
    result = kernel.run(
        task=task_text,
        spec=spec,
        session_id=session,
        base_ref=base_ref,
    )
    _print_result(result)


@main.command("log")
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
)
@click.argument("session")
def log_cmd(workspace: Path, session: str) -> None:
    """Print the checkpoint log for SESSION."""
    memory = Memory(workspace, f"taste/session-{session}")
    history = memory.log()
    table = Table(title=f"Session {session} — {len(history)} checkpoints")
    table.add_column("step", style="cyan")
    table.add_column("sha", style="magenta")
    table.add_column("message")
    for cp in reversed(history):
        table.add_row(cp.step_id, cp.short_sha, cp.message)
    console.print(table)


# -------------------------------------------------------------- rendering

_EVENT_STYLE = {
    "run.start": ("cyan", ">>"),
    "plan.ready": ("green", "PLAN"),
    "step.begin": ("yellow", "STEP"),
    "worker.done": ("blue", "WORK"),
    "monitor.verdict": (None, "EVAL"),  # colored by pass/fail
    "step.rollback": ("red", "REV "),
    "run.halt": ("red bold", "HALT"),
    "run.done": ("green bold", "DONE"),
}


def _print_event(event: Event) -> None:
    style, tag = _EVENT_STYLE.get(event.kind, ("white", event.kind.upper()))
    if event.kind == "monitor.verdict":
        style = "green" if event.payload.get("passed") else "red"
    body = " ".join(f"[dim]{k}=[/]{v}" for k, v in event.payload.items() if k != "evidence")
    console.print(f"[{style}]{tag}[/] {body}")


def _print_result(result: RunResult) -> None:
    console.rule(f"[bold]{result.summary()}[/]")
    table = Table(title="Step outcomes")
    table.add_column("step", style="cyan")
    table.add_column("attempts", justify="right")
    table.add_column("rolled back", justify="center")
    table.add_column("verdict")
    table.add_column("sha")
    for o in result.outcomes:
        verdict_cell = (
            f"[green]PASS[/] {o.verdict.reason}"
            if o.verdict.passed
            else f"[red]FAIL[/] {o.verdict.reason}"
        )
        table.add_row(
            o.step.id,
            str(o.attempts),
            "yes" if o.rolled_back else "",
            verdict_cell,
            o.checkpoint.short_sha,
        )
    console.print(table)
    console.print(
        Panel.fit(
            f"Branch: [cyan]{result.branch}[/]\n"
            f"Inspect: [dim]git -C <workspace> log {result.branch} --oneline[/]\n"
            f"Replay:  [dim]taste log {result.session_id}[/]",
            border_style="dim",
        )
    )


if __name__ == "__main__":
    main()
