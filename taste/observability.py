"""What actually happened during a run, in a form you can look at.

The kernel already emits everything worth seeing — per-role models, the wave
structure, worktree branches opening and merging, Monitor verdicts, recovery
diagnoses and the actions they chose, guard vetoes. None of it was ever
assembled into one picture, so the only way to answer "what were the workers
doing when this broke?" was to read a JSONL by eye.

This module builds that picture. It normalises the event stream into a small
trace model and renders it as inline SVG — no server, no JavaScript, no
external assets, so the output survives being committed to a PR, opened
offline, or dropped into a talk.

**The view that matters most is the regression timeline.** Every other panel
describes what the harness believed; that one shows what was *true*. Laying
the observation spine, the probe verdicts and the Monitor's verdicts on a
single time axis makes a silent regression directly visible: the moment a
previously-passing test goes red, and the Monitor reporting PASS straight
across it. That picture is the paper's claim, and it is much harder to argue
with than a table of rates.

Nothing here is on the measurement path. It reads artifacts after the fact and
computes no number that any result depends on.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- palette

OK = "#2f9e44"
FAIL = "#e03131"
WARN = "#f08c00"
INFO = "#1971c2"
MUTED = "#868e96"
RECOVER = "#7048e8"

_LANE_ORDER = ("kernel", "planner", "worker", "monitor", "recovery", "guard", "merge")

# Which lane each event kind belongs on. A run is a set of concurrent threads,
# and collapsing them into one chronological list is exactly what hides the
# concurrency the architecture is about.
_LANE_OF = {
    "run": "kernel", "wave": "kernel", "journal": "kernel", "shadow": "kernel",
    "plan": "planner",
    "step": "worker", "worker": "worker",
    "monitor": "monitor",
    "recovery": "recovery",
    "guard": "guard",
    "merge": "merge", "worktree": "merge",
}


@dataclass
class ModelUse:
    role: str
    model: str
    temperature: float | None = None


@dataclass
class Mark:
    """One thing that happened, placed on a lane at a time."""

    lane: str
    t: float
    kind: str
    label: str
    colour: str = MUTED
    step_id: str = ""


@dataclass
class StepView:
    step_id: str
    description: str = ""
    depends_on: tuple[str, ...] = ()
    wave: int = 0
    attempts: int = 0
    passed: bool | None = None
    rolled_back: bool = False
    recovery_actions: tuple[str, ...] = ()


@dataclass
class Observation:
    seq: int
    sha: str = ""
    step_id: str = ""
    attempt: int = 0
    trigger: str = ""
    files: tuple[str, ...] = ()
    cost_work_usd: float = 0.0


@dataclass
class Episode:
    probe: str
    onset_seq: int
    recovered_seq: int | None = None
    detected_seq_attributed: int | None = None
    detected_seq_any: int | None = None

    @property
    def silent(self) -> bool:
        return self.detected_seq_attributed is None


@dataclass
class BranchView:
    name: str
    step_id: str = ""
    outcome: str = "open"          # open | merged | conflict


@dataclass
class Trace:
    session: str = ""
    task: str = ""
    branch: str = ""
    status: str = "running"
    harness_sha: str = ""
    models: list[ModelUse] = field(default_factory=list)
    steps: list[StepView] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    branches: list[BranchView] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    verdict_seqs: list[tuple[int, bool]] = field(default_factory=list)
    """(observation seq, passed) for each Monitor verdict that could be placed
    on the timeline. Joined on (step_id, attempt) — see taste.attribution."""
    t0: float = 0.0
    t1: float = 0.0

    @property
    def lanes(self) -> list[str]:
        present = {m.lane for m in self.marks}
        return [lane for lane in _LANE_ORDER if lane in present]

    @property
    def duration(self) -> float:
        return max(self.t1 - self.t0, 1e-6)


# ---------------------------------------------------------------- building


def build_trace(
    events: list[dict[str, Any]],
    *,
    plan: dict[str, Any] | None = None,
    shadow: list[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
) -> Trace:
    """Normalise a run's artifacts into one trace.

    Deliberately tolerant: a run that crashed mid-flight, or one from an older
    harness with a narrower event vocabulary, still renders. A dashboard that
    refuses to draw the run you most need to look at is worthless.
    """
    trace = Trace()
    if events:
        trace.t0 = float(events[0].get("ts", 0.0) or 0.0)
        trace.t1 = float(events[-1].get("ts", 0.0) or 0.0) or trace.t0

    steps: dict[str, StepView] = {}
    for raw in (plan or {}).get("steps", []):
        sid = str(raw.get("id", ""))
        steps[sid] = StepView(
            step_id=sid,
            description=str(raw.get("description", "")),
            depends_on=tuple(raw.get("depends_on", ()) or ()),
        )

    wave_index = 0
    for event in events:
        kind = str(event.get("kind", ""))
        payload = event.get("payload") or {}
        ts = float(event.get("ts", 0.0) or 0.0)
        family = kind.split(".", 1)[0]
        lane = _LANE_OF.get(family, "kernel")
        sid = str(payload.get("id") or payload.get("step") or "")

        if kind == "run.start":
            trace.session = str(payload.get("session", ""))
            trace.task = str(payload.get("task", ""))
            trace.branch = str(payload.get("branch", ""))
        elif kind == "run.manifest":
            trace.harness_sha = str(payload.get("harness_git_sha", ""))
            temps = payload.get("temperature") or {}
            for role, model in (payload.get("models") or {}).items():
                trace.models.append(
                    ModelUse(role=str(role), model=str(model), temperature=temps.get(role))
                )
        elif kind in ("run.done", "run.halt"):
            trace.status = str(payload.get("status", "halted"))
        elif kind == "wave.begin":
            wave_index += 1
            for member in payload.get("steps", ()) or ():
                steps.setdefault(str(member), StepView(step_id=str(member))).wave = wave_index
        elif kind == "step.begin":
            view = steps.setdefault(sid, StepView(step_id=sid))
            view.attempts = max(view.attempts, int(payload.get("attempt", 1) or 1))
        elif kind == "monitor.verdict":
            passed = bool(payload.get("passed"))
            steps.setdefault(sid, StepView(step_id=sid)).passed = passed
        elif kind == "step.rollback":
            steps.setdefault(sid, StepView(step_id=sid)).rolled_back = True
        elif kind == "recovery.action":
            view = steps.setdefault(sid, StepView(step_id=sid))
            view.recovery_actions = (*view.recovery_actions, str(payload.get("action", "?")))
        elif kind == "worktree.open":
            trace.branches.append(
                BranchView(name=str(payload.get("branch", "?")), step_id=sid)
            )
        elif kind in ("worktree.merge", "merge.done"):
            _close_branch(trace, sid, "merged")
        elif kind in ("worktree.conflict", "merge.conflict"):
            _close_branch(trace, sid, "conflict")

        trace.marks.append(
            Mark(
                lane=lane, t=ts, kind=kind, step_id=sid,
                label=_label_for(kind, payload), colour=_colour_for(kind, payload),
            )
        )

    trace.steps = sorted(steps.values(), key=lambda s: (s.wave, s.step_id))

    for row in shadow or []:
        trace.observations.append(
            Observation(
                seq=int(row.get("seq", 0) or 0),
                sha=str(row.get("sha", "")),
                step_id=str(row.get("step_id", "")),
                attempt=int(row.get("attempt", 0) or 0),
                trigger=str(row.get("trigger", "")),
                files=tuple(row.get("files", ()) or ()),
                cost_work_usd=float(row.get("cost_work_usd", 0.0) or 0.0),
            )
        )
    trace.observations.sort(key=lambda o: o.seq)

    for row in (evidence or {}).get("episodes", []):
        trace.episodes.append(
            Episode(
                probe=str(row.get("probe", "")),
                onset_seq=int(row.get("onset_seq", 0) or 0),
                recovered_seq=row.get("recovered_seq"),
                detected_seq_attributed=row.get("detected_seq_attributed"),
                detected_seq_any=row.get("detected_seq_any"),
            )
        )

    trace.verdict_seqs = _place_verdicts(events, trace.observations)
    return trace


def _close_branch(trace: Trace, step_id: str, outcome: str) -> None:
    for branch in reversed(trace.branches):
        if branch.step_id == step_id and branch.outcome == "open":
            branch.outcome = outcome
            return


def _place_verdicts(
    events: list[dict[str, Any]], observations: list[Observation]
) -> list[tuple[int, bool]]:
    """Put each Monitor verdict on the observation whose tree it graded.

    Same join as :func:`taste.attribution.harness_failures`, and exact for the
    same reason: the observation is written between the worker finishing and
    the Monitor evaluating, so ``(step_id, attempt)`` identifies the tree.
    """
    index = {(o.step_id, o.attempt): o.seq for o in observations}
    placed: list[tuple[int, bool]] = []
    for event in events:
        if event.get("kind") != "monitor.verdict":
            continue
        payload = event.get("payload") or {}
        seq = index.get((str(payload.get("id", "")), int(payload.get("attempt", 0) or 0)))
        if seq is not None:
            placed.append((seq, bool(payload.get("passed"))))
    return placed


def _label_for(kind: str, payload: dict[str, Any]) -> str:
    if kind == "monitor.verdict":
        return f"{payload.get('id', '')} {'PASS' if payload.get('passed') else 'FAIL'}"
    if kind == "recovery.action":
        return f"{payload.get('id', '')} -> {payload.get('action', '?')}"
    if kind == "recovery.diagnosis":
        return f"{payload.get('id', '')} {payload.get('failure_class', '?')}"
    if kind == "step.begin":
        return f"{payload.get('id', '')} #{payload.get('attempt', 1)}"
    if kind == "guard.veto":
        return f"veto {payload.get('tool', '')}"
    if kind in ("worktree.open", "worktree.merge"):
        return str(payload.get("branch", ""))
    return str(payload.get("id", "") or kind.split(".", 1)[-1])


def _colour_for(kind: str, payload: dict[str, Any]) -> str:
    if kind == "monitor.verdict":
        return OK if payload.get("passed") else FAIL
    if kind.startswith("recovery"):
        return RECOVER
    if kind.startswith("guard") or kind in ("worktree.conflict", "merge.conflict"):
        return WARN
    if kind in ("run.halt", "wave.halt", "step.abort", "plan.error"):
        return FAIL
    if kind in ("run.done", "wave.done", "merge.done"):
        return OK
    return INFO


# ---------------------------------------------------------------- rendering


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def render_dag(trace: Trace, *, width: int = 900) -> str:
    """The plan as a graph: waves left to right, dependencies as edges."""
    if not trace.steps:
        return "<p class='empty'>No plan recorded.</p>"

    waves: dict[int, list[StepView]] = {}
    for step in trace.steps:
        waves.setdefault(step.wave, []).append(step)
    columns = sorted(waves)

    box_w, box_h, gap_x, gap_y, pad = 168, 46, 62, 22, 16
    height = pad * 2 + max(len(v) for v in waves.values()) * (box_h + gap_y)
    span_x = max(box_w + gap_x, (width - pad * 2 - box_w) // max(len(columns), 1))

    pos: dict[str, tuple[int, int]] = {}
    for ci, wave in enumerate(columns):
        for ri, step in enumerate(waves[wave]):
            pos[step.step_id] = (pad + ci * span_x, pad + ri * (box_h + gap_y))

    edges = []
    for step in trace.steps:
        if step.step_id not in pos:
            continue
        x2, y2 = pos[step.step_id]
        for parent in step.depends_on:
            if parent not in pos:
                continue
            x1, y1 = pos[parent]
            edges.append(
                f'<path d="M{x1 + box_w} {y1 + box_h // 2} '
                f'C{x1 + box_w + 30} {y1 + box_h // 2}, {x2 - 30} {y2 + box_h // 2}, '
                f'{x2} {y2 + box_h // 2}" fill="none" stroke="{MUTED}" '
                f'stroke-width="1.5" marker-end="url(#arrow)"/>'
            )

    boxes = []
    for step in trace.steps:
        if step.step_id not in pos:
            continue
        x, y = pos[step.step_id]
        colour = OK if step.passed else (FAIL if step.passed is False else MUTED)
        badge = ""
        if step.rolled_back:
            badge = f'<text x="{x + box_w - 8}" y="{y + 16}" text-anchor="end" font-size="10" fill="{RECOVER}">rollback</text>'
        elif step.attempts > 1:
            badge = f'<text x="{x + box_w - 8}" y="{y + 16}" text-anchor="end" font-size="10" fill="{WARN}">x{step.attempts}</text>'
        boxes.append(
            f'<g><rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="7" '
            f'fill="var(--card)" stroke="{colour}" stroke-width="2"/>'
            f'<text x="{x + 10}" y="{y + 18}" font-size="12" font-weight="600" fill="var(--fg)">{_esc(step.step_id)}</text>'
            f'<text x="{x + 10}" y="{y + 34}" font-size="10.5" fill="var(--fg-dim)">{_esc(step.description[:26])}</text>'
            f"{badge}</g>"
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
        f'aria-label="plan dependency graph">'
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs>'
        + "".join(edges) + "".join(boxes) + "</svg>"
    )


def render_lanes(trace: Trace, *, width: int = 900) -> str:
    """The htop view: one row per concurrent thread, time along x.

    Collapsing these into a single chronological list is precisely what hides
    the concurrency the architecture exists to provide — a worker branch, the
    Monitor judging it, and a recovery handler deciding what to do next are
    three different threads, and the interesting moments are where they meet.
    """
    lanes = trace.lanes
    if not lanes:
        return "<p class='empty'>No events recorded.</p>"

    label_w, row_h, pad = 88, 30, 12
    plot_w = width - label_w - pad * 2
    height = pad * 2 + len(lanes) * row_h
    y_of = {lane: pad + i * row_h for i, lane in enumerate(lanes)}

    parts = []
    for lane in lanes:
        y = y_of[lane]
        parts.append(
            f'<text x="{pad}" y="{y + row_h // 2 + 4}" font-size="11" font-weight="600" '
            f'fill="var(--fg-dim)">{_esc(lane)}</text>'
            f'<line x1="{label_w}" y1="{y + row_h // 2}" x2="{width - pad}" '
            f'y2="{y + row_h // 2}" stroke="var(--rule)" stroke-width="1"/>'
        )

    for mark in trace.marks:
        frac = (mark.t - trace.t0) / trace.duration
        x = label_w + frac * plot_w
        y = y_of.get(mark.lane)
        if y is None:
            continue
        cy = y + row_h // 2
        title = f"+{mark.t - trace.t0:.1f}s  {mark.kind}  {mark.label}"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{cy}" r="4.5" fill="{mark.colour}" '
            f'fill-opacity="0.9"><title>{_esc(title)}</title></circle>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
        f'aria-label="execution lanes over time">' + "".join(parts) + "</svg>"
    )


def render_regression_timeline(trace: Trace, *, width: int = 900) -> str:
    """Observations, what was true at each, and what the harness said.

    The one view that shows the study's subject directly. A silent regression
    is the shape where a red band opens on the truth row and the Monitor row
    stays green straight across it.
    """
    obs = trace.observations
    if not obs:
        return (
            "<p class='empty'>No shadow observations — this run was not measured, "
            "or the timeline was not loaded.</p>"
        )

    pad, spine_y, mon_y, band_top = 16, 74, 108, 30
    plot_w = width - pad * 2
    n = max(len(obs), 1)
    step_x = plot_w / max(n - 1, 1) if n > 1 else 0
    seq_to_x = {o.seq: pad + i * step_x for i, o in enumerate(obs)}
    height = 150 + max(len(trace.episodes) - 1, 0) * 16

    parts = [
        f'<text x="{pad}" y="16" font-size="11" font-weight="600" fill="var(--fg-dim)">'
        f'ground truth (probe replay)</text>',
        f'<line x1="{pad}" y1="{spine_y}" x2="{width - pad}" y2="{spine_y}" '
        f'stroke="var(--rule)" stroke-width="2"/>',
    ]

    # Episode bands: onset to recovery, or to the end if never repaired.
    for i, ep in enumerate(trace.episodes):
        x1 = seq_to_x.get(ep.onset_seq, pad)
        close = ep.recovered_seq if ep.recovered_seq is not None else obs[-1].seq
        x2 = seq_to_x.get(close, width - pad)
        y = band_top + i * 16
        colour = FAIL if ep.recovered_seq is None else WARN
        parts.append(
            f'<rect x="{x1:.1f}" y="{y}" width="{max(x2 - x1, 3):.1f}" height="11" rx="3" '
            f'fill="{colour}" fill-opacity="{0.75 if ep.silent else 0.35}">'
            f'<title>{_esc(ep.probe)} — onset {ep.onset_seq}, '
            f'{"never recovered" if ep.recovered_seq is None else f"recovered at {ep.recovered_seq}"}, '
            f'{"SILENT" if ep.silent else f"detected at {ep.detected_seq_attributed}"}</title></rect>'
        )
        if ep.recovered_seq is not None:
            parts.append(
                f'<circle cx="{x2:.1f}" cy="{y + 5}" r="4" fill="{OK}"><title>recovered</title></circle>'
            )

    # The observation spine.
    for o in obs:
        x = seq_to_x[o.seq]
        parts.append(
            f'<circle cx="{x:.1f}" cy="{spine_y}" r="4" fill="{INFO}">'
            f'<title>obs {o.seq} — {o.step_id} attempt {o.attempt} ({o.trigger}), '
            f'{len(o.files)} files, ${o.cost_work_usd:.4f} work</title></circle>'
        )

    # What the harness thought, on its own row.
    parts.append(
        f'<text x="{pad}" y="{mon_y - 14}" font-size="11" font-weight="600" '
        f'fill="var(--fg-dim)">what the Monitor reported</text>'
    )
    for seq, passed in trace.verdict_seqs:
        x = seq_to_x.get(seq)
        if x is None:
            continue
        parts.append(
            f'<rect x="{x - 4:.1f}" y="{mon_y}" width="9" height="9" rx="2" '
            f'fill="{OK if passed else FAIL}"><title>obs {seq}: Monitor '
            f'{"PASS" if passed else "FAIL"}</title></rect>'
        )

    silent = sum(1 for e in trace.episodes if e.silent)
    parts.append(
        f'<text x="{pad}" y="{height - 8}" font-size="11" fill="var(--fg-dim)">'
        f'{len(obs)} observations &#183; {len(trace.episodes)} regression '
        f'episode(s) &#183; {silent} silent</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
        f'aria-label="regression timeline">' + "".join(parts) + "</svg>"
    )


def render_models(trace: Trace) -> str:
    if not trace.models:
        return "<p class='empty'>No model manifest recorded.</p>"
    rows = "".join(
        f"<tr><td>{_esc(m.role)}</td><td class='mono'>{_esc(m.model)}</td>"
        f"<td class='mono'>{'—' if m.temperature is None else _esc(m.temperature)}</td></tr>"
        for m in trace.models
    )
    return (
        "<table><thead><tr><th>role</th><th>model</th><th>temperature</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_branches(trace: Trace) -> str:
    if not trace.branches:
        return "<p class='empty'>No worktree branches — steps ran on the session branch.</p>"
    colour = {"merged": OK, "conflict": WARN, "open": MUTED}
    rows = "".join(
        f"<tr><td class='mono'>{_esc(b.name)}</td><td class='mono'>{_esc(b.step_id)}</td>"
        f"<td><span class='pill' style='background:{colour.get(b.outcome, MUTED)}'>"
        f"{_esc(b.outcome)}</span></td></tr>"
        for b in trace.branches
    )
    return (
        "<table><thead><tr><th>branch</th><th>step</th><th>outcome</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def load_shadow(gitdir: Path, session: str = "") -> list[dict[str, Any]]:
    """Read shadow.jsonl if a run was measured. Absent is normal, not an error."""
    path = Path(gitdir) / "shadow.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not session or row.get("session") == session:
            rows.append(row)
    return rows
