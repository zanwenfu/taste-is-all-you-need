"""Render a self-contained HTML dashboard from a kernel run's artifacts.

The dashboard is the ``htop for agents`` piece from the blog. Inputs are the
files the Kernel already writes:

    <workspace>/.taste/plan.json            (tracked; the audit trail)
    <workspace>/.taste/monitor/step-XX.json (tracked; one per step)
    <workspace>/.git/taste/events.jsonl     (untracked; the runtime trace —
                                             lives here so rollbacks cannot
                                             wipe it out)

Plus `git log` on the session branch. Output is one HTML file — no server,
no JS bundles, no external assets. Opens in any browser, commits cleanly
into a PR, screenshots well for talks and docs.
"""

from __future__ import annotations

import html
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taste import observability as obs


@dataclass
class RunArtifacts:
    workspace: Path
    branch: str
    plan: dict[str, Any]
    events: list[dict[str, Any]]
    monitor: dict[str, dict[str, Any]]
    git_log: list[tuple[str, str]]  # (sha, subject)
    shadow: list[dict[str, Any]] = field(default_factory=list)
    """Observations, when the run was measured. Absent is normal."""
    evidence: dict[str, Any] = field(default_factory=dict)
    """The replay sidecar: episodes and the silence report, if scored."""

    @classmethod
    def load(cls, workspace: Path, branch: str | None = None) -> RunArtifacts:
        workspace = Path(workspace).resolve()
        taste_dir = workspace / ".taste"

        plan_path = taste_dir / "plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}

        # Events live in .git/taste/events.jsonl (runtime trace, survives rollback).
        # Old runs may have them in .taste/events.jsonl — read whichever exists.
        events_path = workspace / ".git" / "taste" / "events.jsonl"
        if not events_path.exists():
            events_path = taste_dir / "events.jsonl"
        events: list[dict[str, Any]] = []
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        monitor: dict[str, dict[str, Any]] = {}
        monitor_dir = taste_dir / "monitor"
        if monitor_dir.is_dir():
            for f in sorted(monitor_dir.glob("*.json")):
                monitor[f.stem] = json.loads(f.read_text())

        # Resolve the session branch from the run.start event if not given.
        resolved_branch = branch
        if resolved_branch is None:
            for e in events:
                if e["kind"] == "run.start":
                    resolved_branch = e["payload"].get("branch")
                    break

        git_log: list[tuple[str, str]] = []
        if resolved_branch:
            proc = subprocess.run(
                ["git", "log", "--oneline", "--no-decorate", resolved_branch],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            for line in proc.stdout.splitlines():
                if " " in line:
                    sha, subject = line.split(" ", 1)
                    git_log.append((sha, subject))

        shadow = obs.load_shadow(workspace / ".git" / "taste")
        # The sidecar is written next to the ledger by the sweep driver; when a
        # run was executed standalone there simply is not one.
        evidence: dict[str, Any] = {}
        for candidate in (taste_dir / "evidence.json", workspace / "evidence.json"):
            if candidate.exists():
                evidence = json.loads(candidate.read_text())
                break

        return cls(
            workspace=workspace,
            branch=resolved_branch or "(unknown)",
            plan=plan,
            events=events,
            monitor=monitor,
            git_log=git_log,
            shadow=shadow,
            evidence=evidence,
        )


def render(artifacts: RunArtifacts) -> str:
    """Produce a self-contained HTML string."""
    start = next((e for e in artifacts.events if e["kind"] == "run.start"), {})
    done = next((e for e in reversed(artifacts.events) if e["kind"] == "run.done"), {})

    session = start.get("payload", {}).get("session", "(unknown)")
    task = start.get("payload", {}).get("task", "")
    agent = start.get("payload", {}).get("agent", "")
    status = done.get("payload", {}).get("status", "running")
    elapsed = done.get("payload", {}).get("elapsed", 0.0)
    cost = done.get("payload", {}).get("cost_usd", 0.0)
    cache_rate = done.get("payload", {}).get("cache_hit_rate", 0.0)

    step_summary = _summarize_steps(artifacts)
    trace = obs.build_trace(
        artifacts.events, plan=artifacts.plan,
        shadow=artifacts.shadow, evidence=artifacts.evidence,
    )
    t0 = artifacts.events[0]["ts"] if artifacts.events else 0.0

    return _HTML_TEMPLATE.format(
        session=html.escape(session),
        task=html.escape(task),
        agent=html.escape(agent),
        branch=html.escape(artifacts.branch),
        status=html.escape(status),
        status_class=_status_class(status),
        elapsed=f"{elapsed:.2f}",
        cost=f"{cost:.4f}",
        cache_rate=f"{cache_rate * 100:.1f}%",
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        steps_rows=_render_step_rows(step_summary),
        timeline_items=_render_timeline(artifacts.events, t0),
        git_log_rows=_render_git_log(artifacts.git_log, step_summary),
        events_count=len(artifacts.events),
        steps_count=len(artifacts.plan.get("steps", [])),
        passed_count=sum(1 for s in step_summary.values() if s["passed"]),
        rollback_count=sum(1 for s in step_summary.values() if s["rolled_back"]),
        dag_svg=obs.render_dag(trace),
        lanes_svg=obs.render_lanes(trace),
        regression_svg=obs.render_regression_timeline(trace),
        models_table=obs.render_models(trace),
        branches_table=obs.render_branches(trace),
        episode_count=len(trace.episodes),
        silent_count=sum(1 for e in trace.episodes if e.silent),
        observation_count=len(trace.observations),
    )


def write(
    workspace: Path,
    *,
    output: Path | None = None,
    branch: str | None = None,
) -> Path:
    """Build the dashboard and write it to ``output`` (default: <ws>/.taste/dashboard.html)."""
    artifacts = RunArtifacts.load(workspace, branch=branch)
    out = output or (Path(workspace).resolve() / ".taste" / "dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(artifacts))
    return out


# ================================================================= derived


def _summarize_steps(a: RunArtifacts) -> dict[str, dict[str, Any]]:
    """Per-step: attempts, rollbacks, latest verdict, sha."""
    steps = {
        s["id"]: {
            "description": s["description"],
            "verification": s["verification"],
            "attempts": 0,
            "rolled_back": False,
            "passed": False,
            "reason": "",
            "sha": "",
        }
        for s in a.plan.get("steps", [])
    }
    for event in a.events:
        p = event.get("payload", {})
        sid = p.get("id")
        if sid not in steps:
            continue
        if event["kind"] == "step.begin":
            steps[sid]["attempts"] = max(steps[sid]["attempts"], p.get("attempt", 0))
        elif event["kind"] == "step.rollback":
            steps[sid]["rolled_back"] = True
        elif event["kind"] == "monitor.verdict":
            steps[sid]["passed"] = bool(p.get("passed"))
            steps[sid]["reason"] = p.get("reason", "")
            steps[sid]["sha"] = p.get("sha", "")
    return steps


# ================================================================= rendering


def _status_class(status: str) -> str:
    return {"completed": "pass", "failed": "fail", "running": "info"}.get(status, "info")


def _kind_class(kind: str) -> str:
    if kind.startswith("step.rollback") or kind == "run.halt" or kind == "plan.error":
        return "rollback"
    if kind == "monitor.verdict":
        return "monitor"
    if kind == "worker.done":
        return "worker"
    if kind == "plan.ready":
        return "plan"
    if kind in ("run.start", "run.done"):
        return "lifecycle"
    return "other"


def _fmt_payload(payload: dict[str, Any]) -> str:
    bits = []
    for k, v in payload.items():
        if k in ("evidence", "task", "criteria"):
            continue
        s = str(v)
        if len(s) > 120:
            s = s[:117] + "…"
        bits.append(f"{k}=<b>{html.escape(s)}</b>")
    return " ".join(bits)


def _render_step_rows(steps: dict[str, dict[str, Any]]) -> str:
    if not steps:
        return "<tr><td colspan='6'><em>No plan yet.</em></td></tr>"
    rows = []
    for sid, s in steps.items():
        badge = "PASS" if s["passed"] else "FAIL"
        badge_class = "pass" if s["passed"] else "fail"
        rollback = "yes" if s["rolled_back"] else ""
        verification = s["verification"]
        vcell = (
            f"<code>{html.escape(verification.get('command') or '')}</code>"
            if verification["kind"] == "shell"
            else f"<em>llm: {html.escape(verification.get('criteria') or '')}</em>"
        )
        rows.append(
            f"<tr>"
            f"<td><code>{html.escape(sid)}</code></td>"
            f"<td>{html.escape(s['description'])}</td>"
            f"<td>{vcell}</td>"
            f"<td class='center'>{s['attempts']}</td>"
            f"<td class='center rollback-cell'>{rollback}</td>"
            f"<td><span class='badge badge-{badge_class}'>{badge}</span> "
            f"<span class='reason'>{html.escape(s['reason'])}</span></td>"
            f"<td><code>{html.escape(s['sha'])}</code></td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _render_timeline(events: list[dict[str, Any]], t0: float) -> str:
    items = []
    for e in events:
        dt = e["ts"] - t0
        cls = _kind_class(e["kind"])
        items.append(
            f"<li class='evt evt-{cls}'>"
            f"<span class='ts'>+{dt:.2f}s</span>"
            f"<span class='kind'>{html.escape(e['kind'])}</span>"
            f"<span class='payload'>{_fmt_payload(e.get('payload', {}))}</span>"
            f"</li>"
        )
    return "\n".join(items) if items else "<li><em>No events.</em></li>"


def _render_git_log(log: list[tuple[str, str]], steps: dict[str, dict[str, Any]]) -> str:
    step_shas = {s["sha"]: sid for sid, s in steps.items() if s["sha"]}
    if not log:
        return "<tr><td colspan='3'><em>No commits (run the kernel first).</em></td></tr>"
    rows = []
    for sha, subject in log:
        short = sha[:7]
        step_tag = ""
        for ssha, sid in step_shas.items():
            if sha.startswith(ssha):
                step_tag = f"<span class='step-tag'>{sid}</span>"
                break
        rows.append(
            f"<tr>"
            f"<td><code>{html.escape(short)}</code></td>"
            f"<td>{step_tag}</td>"
            f"<td>{html.escape(subject)}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


# ================================================================= template


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>taste dashboard — session {session}</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
  background: #0d1117; color: #e6edf3; margin: 0; padding: 24px 32px;
  line-height: 1.5;
}}
a {{ color: #58a6ff; }}
code {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.92em;
        background: #1f242d; padding: 2px 6px; border-radius: 4px; color: #d2a8ff; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; text-transform: uppercase; letter-spacing: 0.08em;
      color: #8b949e; margin: 32px 0 12px; border-bottom: 1px solid #21262d;
      padding-bottom: 6px; }}
section {{ margin-bottom: 24px; }}
.meta {{ color: #8b949e; font-size: 13px; margin-bottom: 16px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 12px; margin: 16px 0 0; }}
.card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px;
         padding: 14px 16px; }}
.card .label {{ color: #8b949e; font-size: 11px; text-transform: uppercase;
                letter-spacing: 0.08em; }}
.card .value {{ font-size: 22px; font-weight: 600; margin-top: 6px; font-variant-numeric: tabular-nums; }}
.card.pass .value {{ color: #3fb950; }}
.card.fail .value {{ color: #f85149; }}
.card.info .value {{ color: #58a6ff; }}
.card.danger .value {{ color: #f85149; }}

/* Panels added by taste.observability. The SVGs reference these variables so
   the drawing inherits the page theme rather than hard-coding two palettes. */
:root {{
  --card: #161b22;
  --fg: #e6edf3;
  --fg-dim: #8b949e;
  --rule: #30363d;
}}
.note {{ color: #8b949e; font-size: 12.5px; margin: 0 0 10px; max-width: 78ch; }}
.note em {{ color: #e6edf3; font-style: normal; font-weight: 600; }}
.empty {{ color: #6e7681; font-size: 13px; font-style: italic; }}
svg {{ display: block; max-width: 100%; overflow: visible; }}
.split {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; }}
.split h2 {{ margin-top: 0; }}
.mono {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }}
.pill {{ display: inline-block; padding: 1px 8px; border-radius: 10px; color: #fff;
         font-size: 11px; font-weight: 600; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #21262d;
         vertical-align: top; }}
th {{ color: #8b949e; font-weight: 500; text-transform: uppercase;
      letter-spacing: 0.06em; font-size: 11px; }}
.center {{ text-align: center; }}
.badge {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px;
          border-radius: 4px; }}
.badge-pass {{ background: #1b3c21; color: #3fb950; }}
.badge-fail {{ background: #4d1d1d; color: #f85149; }}
.reason {{ color: #8b949e; font-size: 12px; }}
.rollback-cell {{ color: #d29922; font-weight: 600; }}
.step-tag {{ font-size: 11px; color: #d2a8ff; background: #1f242d; padding: 1px 6px;
             border-radius: 4px; letter-spacing: 0.02em; }}
ol.timeline {{ list-style: none; padding-left: 0; margin: 0; border-left: 2px solid #21262d; }}
.evt {{ padding: 6px 0 6px 16px; margin-left: -2px; border-left: 2px solid transparent;
         display: grid; grid-template-columns: 80px 150px 1fr; gap: 12px; font-size: 13px; }}
.evt .ts {{ color: #8b949e; font-family: "SF Mono", Menlo, monospace;
            font-variant-numeric: tabular-nums; }}
.evt .kind {{ font-weight: 600; }}
.evt .payload {{ color: #c9d1d9; }}
.evt-lifecycle {{ border-left-color: #58a6ff; }}
.evt-lifecycle .kind {{ color: #58a6ff; }}
.evt-plan {{ border-left-color: #a371f7; }}
.evt-plan .kind {{ color: #a371f7; }}
.evt-worker {{ border-left-color: #79c0ff; }}
.evt-worker .kind {{ color: #79c0ff; }}
.evt-monitor {{ border-left-color: #3fb950; }}
.evt-monitor .kind {{ color: #3fb950; }}
.evt-rollback {{ border-left-color: #f85149; }}
.evt-rollback .kind {{ color: #f85149; }}
.evt b {{ color: #e6edf3; font-weight: 600; }}
.footer {{ text-align: center; color: #6e7681; font-size: 12px; margin-top: 40px;
           padding-top: 16px; border-top: 1px solid #21262d; }}
</style>
</head>
<body>
<header>
  <h1>session <code>{session}</code></h1>
  <div class="meta">
    agent <code>{agent}</code> &middot;
    branch <code>{branch}</code> &middot;
    task <em>{task}</em>
  </div>
  <div class="cards">
    <div class="card {status_class}"><div class="label">status</div><div class="value">{status}</div></div>
    <div class="card info"><div class="label">steps passed</div><div class="value">{passed_count}/{steps_count}</div></div>
    <div class="card info"><div class="label">rollbacks</div><div class="value">{rollback_count}</div></div>
    <div class="card info"><div class="label">elapsed (s)</div><div class="value">{elapsed}</div></div>
    <div class="card info"><div class="label">cost (USD)</div><div class="value">${cost}</div></div>
    <div class="card info"><div class="label">cache hit rate</div><div class="value">{cache_rate}</div></div>
    <div class="card info"><div class="label">events</div><div class="value">{events_count}</div></div>
    <div class="card info"><div class="label">observations</div><div class="value">{observation_count}</div></div>
    <div class="card info"><div class="label">regressions</div><div class="value">{episode_count}</div></div>
    <div class="card danger"><div class="label">silent</div><div class="value">{silent_count}</div></div>
  </div>
</header>

<section>
  <h2>Ground truth vs what the harness noticed</h2>
  <p class="note">The upper band is what was <em>true</em>, recovered by replaying probes
  the agent never saw. The lower row is what the Monitor <em>reported</em>. A red band with
  green squares across it is a silent regression.</p>
  {regression_svg}
</section>

<section>
  <h2>Workflow</h2>
  <p class="note">Steps by wave; edges are declared dependencies. Border colour is the
  Monitor's verdict.</p>
  {dag_svg}
</section>

<section>
  <h2>Threads</h2>
  <p class="note">One row per concurrent thread. Hover any point for its event.</p>
  {lanes_svg}
</section>

<section class="split">
  <div>
    <h2>Models</h2>
    {models_table}
  </div>
  <div>
    <h2>Branches</h2>
    {branches_table}
  </div>
</section>

<section>
  <h2>Plan &amp; outcomes</h2>
  <table>
    <thead>
      <tr><th>step</th><th>description</th><th>verification</th><th>attempts</th>
          <th>rollback</th><th>verdict</th><th>sha</th></tr>
    </thead>
    <tbody>
      {steps_rows}
    </tbody>
  </table>
</section>

<section>
  <h2>Timeline</h2>
  <ol class="timeline">
    {timeline_items}
  </ol>
</section>

<section>
  <h2>Git topology (session branch)</h2>
  <table>
    <thead>
      <tr><th>sha</th><th>step</th><th>subject</th></tr>
    </thead>
    <tbody>
      {git_log_rows}
    </tbody>
  </table>
</section>

<div class="footer">
  generated {generated_at} by
  <a href="https://github.com/zanwenfu/taste-is-all-you-need">taste</a>
</div>
</body>
</html>
"""
