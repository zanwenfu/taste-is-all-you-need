"""``taste`` CLI — ``taste run <task> --agent <spec>``."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from taste import dashboard as dashboard_mod
from taste import journal as journal_mod
from taste import server as server_mod
from taste import viz as viz_mod
from taste.agent import AgentSpec
from taste.config import HarnessConfig
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
@click.option(
    "--arm",
    default=None,
    help=(
        "Harness configuration to run: A0, A2, A3, A3prime, tiered, full. "
        "Omit for the original kernel with every subsystem off."
    ),
)
def run_cmd(
    task: tuple[str, ...],
    agent_path: Path,
    workspace: Path,
    session: str | None,
    base_ref: str,
    max_retries: int,
    arm: str | None,
) -> None:
    """Run an agent on TASK. Every step becomes a commit on a fresh branch."""
    spec = AgentSpec.from_file(agent_path)
    task_text = " ".join(task)

    try:
        config = HarnessConfig.arm(arm, max_retries=max_retries) if arm else None
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--arm") from exc

    header = (
        f"[bold]{spec.name}[/] — {spec.description}\n"
        f"[dim]workspace:[/] {workspace}\n"
        f"[dim]task:[/] {task_text}"
    )
    if config is not None:
        header += f"\n[dim]harness:[/] {config.label} [dim]({config.hash()})[/]"
    console.print(Panel.fit(header, title="taste run", border_style="cyan"))

    llm = LLM()
    kernel = Kernel(
        workspace=workspace,
        llm=llm,
        max_retries=max_retries,
        on_event=_print_event,
        config=config,
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


@main.command("index")
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
)
@click.option("--yaml", "as_yaml", is_flag=True, help="Render as YAML instead of a table.")
@click.option("--limit", default=None, type=int, help="Show only the last N checkpoints.")
@click.argument("session")
def index_cmd(workspace: Path, session: str, as_yaml: bool, limit: int | None) -> None:
    """Scan SESSION's checkpoints — the cheap read before paging in a diff.

    One `git log` for the whole branch. Use `taste card <sha>` for one
    checkpoint's detail, and plain `git show <sha>` for the full diff.
    """
    memory = Memory(workspace, f"taste/session-{session}")
    index = journal_mod.load_index(memory)

    if as_yaml:
        console.print(index.to_yaml(limit=limit))
        return

    cards = index.cards[-limit:] if limit else index.cards
    table = Table(title=f"Session {session} — {len(index.cards)} checkpoints")
    table.add_column("sha", style="magenta")
    table.add_column("step", style="cyan")
    table.add_column("verdict")
    table.add_column("files", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("intent")
    for card in cards:
        verdict_style = {"pass": "green", "fail": "red"}.get(card.verdict, "dim")
        table.add_row(
            card.sha[:7],
            card.step_id,
            f"[{verdict_style}]{card.verdict}[/]",
            str(len(card.files)) if card.files else "-",
            f"${card.cost_usd:.4f}" if card.cost_usd else "-",
            (card.intent[:60] + "…") if len(card.intent) > 60 else card.intent,
        )
    console.print(table)
    if index.degraded:
        console.print(f"[dim]{index.degraded} checkpoint(s) predate journalling[/]")


@main.command("card")
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
)
@click.option("--session", default=None, help="Session branch to read from.")
@click.argument("sha")
def card_cmd(workspace: Path, sha: str, session: str | None) -> None:
    """Show one checkpoint's card — the node detail, without the full diff."""
    branch = f"taste/session-{session}" if session else "HEAD"
    memory = Memory(workspace, branch)
    card = journal_mod.Journal(
        memory, gitdir=Path(memory.repo.git_dir) / "taste"
    ).read(sha)
    if card is None:
        console.print(f"[yellow]no card for {sha}[/] — try `git show {sha}`")
        raise SystemExit(1)
    console.print(Panel.fit(card.to_yaml_block(), title=f"card {card.sha[:7]}"))


@main.command("serve")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=Path.cwd(), help="Directory to scan for runs.")
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bound to localhost by default: the console has no auth.")
def serve_cmd(root: Path, port: int, host: str) -> None:
    """Live console. Watch runs as they happen, step by step."""
    console.print(
        Panel.fit(
            f"Console:  [cyan]http://{host}:{port}[/]\n"
            f"Watching: [dim]{root.resolve()}[/]\n"
            f"[dim]Ctrl-C to stop.[/]",
            title="taste serve", border_style="cyan",
        )
    )
    server_mod.serve_forever(root, host=host, port=port)


@main.command("report")
@click.option("--workspace", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=Path.cwd(), help="Workspace containing a .taste/ directory.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Where to write the HTML. Defaults to <workspace>/.taste/report.html.")
@click.option("--ledger", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="A sweep ledger directory. Builds the index across all cells.")
@click.option("--no-diffs", is_flag=True, help="Skip per-observation diffs (much smaller file).")
def report_cmd(workspace: Path, output: Path | None, ledger: Path | None, no_diffs: bool) -> None:
    """Interactive HTML report: click an observation to see what changed there."""
    if ledger is not None:
        path = viz_mod.write_index(ledger, output=output)
        label = "sweep index"
    else:
        path = viz_mod.write_run(workspace, output=output, with_diffs=not no_diffs)
        label = "run report"
    console.print(
        Panel.fit(
            f"{label}:  [cyan]{path}[/]\n"
            f"Open with:  [dim]open {path}[/]  (macOS)  |  [dim]xdg-open {path}[/]  (Linux)",
            title="taste report", border_style="cyan",
        )
    )


@main.command("dashboard")
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Workspace containing a .taste/ directory (default: cwd).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write the HTML. Defaults to <workspace>/.taste/dashboard.html.",
)
@click.option("--branch", default=None, help="Session branch name override.")
def dashboard_cmd(workspace: Path, output: Path | None, branch: str | None) -> None:
    """Render a self-contained HTML dashboard from a run's .taste/ artifacts."""
    path = dashboard_mod.write(workspace, output=output, branch=branch)
    console.print(
        Panel.fit(
            f"Dashboard:  [cyan]{path}[/]\n"
            f"Open with:  [dim]open {path}[/]  (macOS)  |  [dim]xdg-open {path}[/]  (Linux)",
            title="taste dashboard",
            border_style="cyan",
        )
    )


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

    if result.stats and result.stats.per_model:
        usage_table = Table(title="Model usage", show_edge=False)
        usage_table.add_column("model", style="magenta")
        usage_table.add_column("calls", justify="right")
        usage_table.add_column("input", justify="right")
        usage_table.add_column("output", justify="right")
        usage_table.add_column("cache read", justify="right")
        usage_table.add_column("cost (USD)", justify="right")
        for model, u in sorted(result.stats.per_model.items()):
            usage_table.add_row(
                model,
                str(u.calls),
                f"{u.input_tokens:,}",
                f"{u.output_tokens:,}",
                f"{u.cache_read_tokens:,}",
                f"${u.cost_usd(model):.4f}",
            )
        totals = result.stats.totals
        usage_table.add_row(
            "[bold]total[/]",
            f"[bold]{totals.calls}[/]",
            f"[bold]{totals.input_tokens:,}[/]",
            f"[bold]{totals.output_tokens:,}[/]",
            f"[bold]{totals.cache_read_tokens:,}[/]",
            f"[bold]${result.stats.total_cost_usd:.4f}[/]",
        )
        console.print(usage_table)
        console.print(
            f"[dim]cache hit rate: {result.stats.cache_hit_rate:.1%}[/]"
        )

    console.print(
        Panel.fit(
            f"Branch:   [cyan]{result.branch}[/]\n"
            f"Inspect:  [dim]git -C <workspace> log {result.branch} --oneline[/]\n"
            f"Replay:   [dim]taste log {result.session_id}[/]\n"
            f"Events:   [dim]<workspace>/.git/taste/events.jsonl[/]",
            border_style="dim",
        )
    )


if __name__ == "__main__":
    main()
