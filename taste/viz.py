"""An interactive report you can actually diagnose a run from.

:mod:`taste.dashboard` renders a fixed picture. That is the right thing for a
screenshot or a PR, and the wrong thing when a real run has gone sideways and
the question is *what did the agent actually do at observation 7*. Answering
that needs the diff at each point, the tool calls behind it, and the verdict
the Monitor gave — none of which fits in a static panel.

So this builds one self-contained HTML file with the run's data embedded as
JSON and a few hundred bytes of vanilla JavaScript over it. No server, no
bundler, no CDN: it opens from local disk, works offline, and survives being
committed. Clicking an observation shows what changed there; the event log
filters by thread; every step expands into its checkpoint card.

Two entry points. :func:`render_run` for one cell, and :func:`render_index`
for a sweep — the index is what you look at first when ten instances have run
and you need to know which one to open.

**Diffs are the one thing that needs care.** A real repository observation can
carry thousands of changed lines, and embedding all of them turns a report
into a browser hang. Each diff is capped and the truncation is stated rather
than hidden, because a silently shortened diff is worse than no diff.
"""

from __future__ import annotations

import html
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from taste import observability as obs

DIFF_LINE_CAP = 400
"""Per-observation diff lines kept. Beyond this the report stops being usable."""


@dataclass
class RunPayload:
    """Everything the page needs, in one JSON blob."""

    session: str = ""
    task: str = ""
    branch: str = ""
    status: str = "unknown"
    harness_sha: str = ""
    arm: str = ""
    instance: str = ""
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    models: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[list[Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    branches: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Things the reader should know about this report's own limits."""


def _diff_between(workspace: Path, old: str | None, new: str) -> tuple[str, bool]:
    """Diff between two shadow commits, capped.

    Returns ``(text, truncated)``. Failure is not an error worth raising: a
    missing object means the chain was gc'd or the workspace moved, and the
    report should say so rather than refuse to render.
    """
    # With no previous observation, diff against the commit's own parent —
    # for the first shadow commit that is the materialised base tree.
    args = ["git", "diff", "--no-color", old or f"{new}^", new]
    try:
        proc = subprocess.run(
            args, cwd=workspace, capture_output=True, text=True, timeout=30, check=False
        )
    except Exception:
        return ("(diff unavailable)", False)
    if proc.returncode != 0:
        return ("(diff unavailable — object missing or workspace moved)", False)
    lines = proc.stdout.splitlines()
    if len(lines) > DIFF_LINE_CAP:
        return ("\n".join(lines[:DIFF_LINE_CAP]), True)
    return ("\n".join(lines), False)


def build_payload(
    workspace: Path,
    *,
    branch: str | None = None,
    evidence: dict[str, Any] | None = None,
    arm: str = "",
    instance: str = "",
    with_diffs: bool = True,
) -> RunPayload:
    """Assemble one run's report data from what it left on disk."""
    from taste.dashboard import RunArtifacts

    workspace = Path(workspace).resolve()
    artifacts = RunArtifacts.load(workspace, branch=branch)
    ev = evidence if evidence is not None else artifacts.evidence
    trace = obs.build_trace(
        artifacts.events, plan=artifacts.plan, shadow=artifacts.shadow, evidence=ev
    )

    payload = RunPayload(
        session=trace.session, task=trace.task, branch=trace.branch,
        status=trace.status, harness_sha=trace.harness_sha,
        arm=arm, instance=instance,
        models=[asdict(m) for m in trace.models],
        verdicts=[[seq, passed] for seq, passed in trace.verdict_seqs],
        branches=[asdict(b) for b in trace.branches],
        episodes=list((ev or {}).get("episodes", [])),
    )

    done = next((e for e in reversed(artifacts.events) if e.get("kind") == "run.done"), {})
    payload.cost_usd = float((done.get("payload") or {}).get("cost_usd", 0.0) or 0.0)
    payload.elapsed_s = float((done.get("payload") or {}).get("elapsed", 0.0) or 0.0)

    verdict_at = dict(trace.verdict_seqs)
    open_spans = [
        (e.get("onset_seq"), e.get("recovered_seq"), e.get("probe"))
        for e in payload.episodes
    ]

    previous: str | None = None
    for o in trace.observations:
        diff, truncated = ("", False)
        if with_diffs:
            diff, truncated = _diff_between(workspace, previous, o.sha)
        broken = [
            probe for onset, recovered, probe in open_spans
            if onset is not None and o.seq >= onset and (recovered is None or o.seq < recovered)
        ]
        payload.observations.append({
            "seq": o.seq, "sha": o.sha, "step_id": o.step_id, "attempt": o.attempt,
            "trigger": o.trigger, "files": list(o.files),
            "cost_work_usd": round(o.cost_work_usd, 6),
            "monitor": verdict_at.get(o.seq),
            "broken": broken,
            "diff": diff, "diff_truncated": truncated,
        })
        previous = o.sha

    cards = _load_cards(workspace, trace.branch)
    for s in trace.steps:
        payload.steps.append({
            "id": s.step_id, "description": s.description,
            "depends_on": list(s.depends_on), "wave": s.wave,
            "attempts": s.attempts, "passed": s.passed,
            "rolled_back": s.rolled_back, "recovery": list(s.recovery_actions),
            "cards": cards.get(s.step_id, []),
        })

    for mark in trace.marks:
        payload.events.append({
            "t": round(mark.t - trace.t0, 3), "kind": mark.kind, "lane": mark.lane,
            "label": mark.label, "step": mark.step_id, "colour": mark.colour,
        })

    if not trace.observations:
        payload.notes.append(
            "No shadow observations: this run was not measured, so no regression "
            "timeline exists."
        )
    if not ev:
        payload.notes.append(
            "No replay sidecar: probes were never replayed, so regressions are "
            "unknown rather than absent."
        )
    if any(o["diff_truncated"] for o in payload.observations):
        payload.notes.append(f"Some diffs truncated at {DIFF_LINE_CAP} lines.")
    return payload


def _load_cards(workspace: Path, branch: str) -> dict[str, list[dict[str, Any]]]:
    """Checkpoint cards per step, when the journal was on. Absent is normal."""
    out: dict[str, list[dict[str, Any]]] = {}
    path = Path(workspace) / ".git" / "taste" / "journal.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.setdefault(str(raw.get("step_id", "")), []).append(raw)
    return out


# ---------------------------------------------------------------- rendering


def _esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def render_run(payload: RunPayload, *, title: str = "") -> str:
    """One self-contained interactive page for a single run."""
    data = json.dumps(asdict(payload)).replace("</", "<\\/")
    heading = title or (payload.instance or payload.session or "run")
    return _PAGE.replace("__TITLE__", _esc(heading)).replace("__DATA__", data)


def render_index(rows: list[dict[str, Any]], *, title: str = "sweep") -> str:
    """The page you open first when many cells have run."""
    data = json.dumps({"rows": rows}).replace("</", "<\\/")
    return _INDEX.replace("__TITLE__", _esc(title)).replace("__DATA__", data)


def write_run(
    workspace: Path,
    *,
    output: Path | None = None,
    evidence: dict[str, Any] | None = None,
    arm: str = "",
    instance: str = "",
    with_diffs: bool = True,
) -> Path:
    payload = build_payload(
        workspace, evidence=evidence, arm=arm, instance=instance, with_diffs=with_diffs
    )
    out = Path(output) if output else Path(workspace) / ".taste" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_run(payload), encoding="utf-8")
    return out


def write_index(ledger_dir: Path, *, output: Path | None = None) -> Path:
    """Scan a sweep ledger and build the overview page."""
    ledger_dir = Path(ledger_dir)
    rows: list[dict[str, Any]] = []
    for cell in sorted(ledger_dir.glob("*.json")):
        try:
            raw = json.loads(cell.read_text())
        except json.JSONDecodeError:
            continue
        evidence_path = ledger_dir / "evidence" / cell.name
        ev = {}
        if evidence_path.exists():
            try:
                ev = json.loads(evidence_path.read_text())
            except json.JSONDecodeError:
                ev = {}
        silence = ev.get("silence") or {}
        rows.append({
            "task": raw.get("task", ""), "arm": raw.get("arm", ""),
            "trial": raw.get("trial", 0), "status": raw.get("status", ""),
            "score": raw.get("score"),
            "billed_usd": raw.get("billed_usd", 0.0),
            "work_usd": raw.get("work_usd", 0.0),
            "elapsed_s": raw.get("elapsed_s", 0.0),
            "steps": f"{raw.get('steps_passed', 0)}/{raw.get('steps_total', 0)}",
            "rollbacks": raw.get("rollbacks", 0),
            "observations": ev.get("observations", 0),
            "episodes": len(ev.get("episodes", [])),
            "silent": silence.get("silent_attributed", 0),
            "report": raw.get("report_path", ""),
            "workspace": raw.get("workspace", ""),
            "error": raw.get("error"),
        })
    out = Path(output) if output else ledger_dir / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_index(rows), encoding="utf-8")
    return out


_CSS = """
:root{--ground:#F5F7F6;--panel:#FFF;--sunk:#EDF1F0;--ink:#0F1619;--dim:#5A686E;
--rule:#DDE4E3;--sig:#1F6F6C;--pass:#2C8B52;--fail:#C43D35;--warn:#B37B12;--rec:#6A4BE0;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--ground:#0B1013;--panel:#131B20;--sunk:#0F1619;--ink:#DCE6E9;--dim:#7C8F97;
--rule:#222F35;--sig:#4FB3AE;--pass:#3FA86A;--fail:#E05A50;--warn:#D2971F;--rec:#8E72FF;}}
:root[data-theme="dark"]{--ground:#0B1013;--panel:#131B20;--sunk:#0F1619;--ink:#DCE6E9;
--dim:#7C8F97;--rule:#222F35;--sig:#4FB3AE;--pass:#3FA86A;--fail:#E05A50;--warn:#D2971F;--rec:#8E72FF;}
*{box-sizing:border-box}
body{margin:0;padding:28px 22px 64px;background:var(--ground);color:var(--ink);
font:14.5px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
.mono,code,pre,th,.tag,.pill,.k{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
h1{font-family:ui-monospace,Menlo,monospace;font-size:clamp(19px,3vw,26px);margin:0 0 6px;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin:0 0 10px;font-family:ui-monospace,Menlo,monospace}
p{margin:0 0 10px;color:var(--dim);max-width:74ch}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:18px 20px}
.scroll{overflow-x:auto}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(116px,1fr));gap:1px;background:var(--rule);
border:1px solid var(--rule);border-radius:10px;overflow:hidden}
.cell{background:var(--panel);padding:12px 14px}
.cell .lab{font:10.5px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
.cell .val{font:600 20px/1.2 ui-monospace,Menlo,monospace;margin-top:4px;font-variant-numeric:tabular-nums}
.cell.alert .val{color:var(--fail)}.cell.good .val{color:var(--pass)}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);
padding:0 12px 7px 0;border-bottom:1px solid var(--rule);font-weight:600;white-space:nowrap}
td{padding:8px 12px 8px 0;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:10.5px;font-weight:600;border:1px solid currentColor}
.pill.pass{color:var(--pass)}.pill.fail{color:var(--fail)}.pill.warn{color:var(--warn)}.pill.idle{color:var(--dim)}
.tag{font-size:10px;padding:1px 6px;border-radius:4px;background:var(--sunk);color:var(--dim);margin-right:4px}
.tag.rec{color:var(--rec)}.tag.warn{color:var(--warn)}
.tag.hk{opacity:.5;font-style:italic}
button{font:inherit;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;cursor:pointer;
background:var(--sunk);color:var(--dim);border:1px solid var(--rule);border-radius:6px;padding:4px 10px}
button[aria-pressed="true"]{background:var(--sig);color:var(--panel);border-color:var(--sig)}
button:focus-visible{outline:2px solid var(--sig);outline-offset:2px}
.row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
pre{background:var(--sunk);border:1px solid var(--rule);border-radius:7px;padding:12px;
overflow-x:auto;font-size:12px;line-height:1.45;margin:0;max-height:420px}
.d-add{color:var(--pass)}.d-del{color:var(--fail)}.d-hd{color:var(--sig);font-weight:600}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.obs{display:flex;flex-wrap:wrap;gap:5px}
.obs button{position:relative;min-width:38px}
.obs button.brk{border-color:var(--fail);color:var(--fail)}
.obs button.mon-pass::after{content:"";position:absolute;right:3px;top:3px;width:5px;height:5px;border-radius:50%;background:var(--pass)}
.obs button.mon-fail::after{content:"";position:absolute;right:3px;top:3px;width:5px;height:5px;border-radius:50%;background:var(--fail)}
.empty{color:var(--dim);font-style:italic;font-size:13px}
.note{background:var(--sunk);border-left:3px solid var(--warn);padding:9px 14px;border-radius:0 6px 6px 0;
font-size:12.5px;color:var(--ink);margin-bottom:8px}
.foot{color:var(--dim);font-size:11.5px;border-top:1px solid var(--rule);padding-top:14px}
a{color:var(--sig)}
"""

_PAGE = """<title>__TITLE__</title>
<style>""" + _CSS + """</style>
<div class="wrap">
  <header>
    <h1 id="h"></h1>
    <p id="sub"></p>
  </header>
  <div class="strip" id="strip"></div>
  <div id="notes"></div>

  <section class="panel">
    <h2>Observations — click one to see what changed</h2>
    <p>Dot in the corner is the Monitor's verdict at that point. A red outline means a
    previously-passing test was broken there.</p>
    <div class="obs" id="obs"></div>
    <div id="detail" style="margin-top:16px"></div>
  </section>

  <section class="panel">
    <h2>Steps</h2>
    <div class="scroll"><table id="steps"></table></div>
  </section>

  <section class="panel">
    <h2>Events</h2>
    <div class="row" id="lanes"></div>
    <div class="scroll"><table id="events"></table></div>
  </section>

  <div class="two">
    <section class="panel"><h2>Models</h2><div class="scroll"><table id="models"></table></div></section>
    <section class="panel"><h2>Branches</h2><div class="scroll"><table id="branches"></table></div></section>
  </div>

  <p class="foot" id="foot"></p>
</div>
<script id="payload" type="application/json">__DATA__</script>
<script>
(function(){
  var D = JSON.parse(document.getElementById('payload').textContent);
  var E = function(s){var d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;};
  var $ = function(id){return document.getElementById(id);};

  $('h').textContent = D.instance || D.session || 'run';
  $('sub').textContent = D.task || '';

  var silent = (D.episodes||[]).filter(function(e){return e.detected_seq_attributed==null;}).length;
  var cells = [
    ['status', D.status, D.status==='completed'?'good':''],
    ['arm', D.arm||'—', ''], ['steps', (D.steps||[]).length, ''],
    ['observations', (D.observations||[]).length, ''],
    ['regressions', (D.episodes||[]).length, ''],
    ['silent', silent, silent>0?'alert':''],
    ['cost', '$'+(D.cost_usd||0).toFixed(4), ''],
    ['elapsed', (D.elapsed_s||0).toFixed(1)+'s', '']
  ];
  $('strip').innerHTML = cells.map(function(c){
    return '<div class="cell '+c[2]+'"><div class="lab">'+E(c[0])+'</div><div class="val">'+E(c[1])+'</div></div>';
  }).join('');

  $('notes').innerHTML = (D.notes||[]).map(function(n){return '<div class="note">'+E(n)+'</div>';}).join('');

  // --- observations
  var obs = D.observations||[];
  $('obs').innerHTML = obs.length ? obs.map(function(o,i){
    var cls = (o.broken&&o.broken.length?'brk ':'') + (o.monitor===true?'mon-pass':(o.monitor===false?'mon-fail':''));
    return '<button class="'+cls+'" data-i="'+i+'" aria-pressed="false">'+o.seq+'</button>';
  }).join('') : '<span class="empty">This run was not measured.</span>';

  function colourDiff(t){
    return t.split('\\n').map(function(l){
      var c = l[0]==='+'&&l.slice(0,3)!=='+++' ? 'd-add' : (l[0]==='-'&&l.slice(0,3)!=='---' ? 'd-del'
            : (l.slice(0,2)==='@@'||l.slice(0,4)==='diff' ? 'd-hd' : ''));
      return c ? '<span class="'+c+'">'+E(l)+'</span>' : E(l);
    }).join('\\n');
  }

  function showObs(i){
    var o = obs[i];
    Array.prototype.forEach.call($('obs').children, function(b){
      b.setAttribute('aria-pressed', b.dataset.i===String(i)?'true':'false');
    });
    var mon = o.monitor===true?'<span class="pill pass">Monitor PASS</span>'
            : (o.monitor===false?'<span class="pill fail">Monitor FAIL</span>'
            : '<span class="pill idle">no verdict here</span>');
    var brk = o.broken&&o.broken.length
      ? '<div class="note" style="border-color:var(--fail)">Broken at this point: '
        + o.broken.map(E).join(', ') + '</div>' : '';
    // .taste/ paths are the harness's own bookkeeping, not the agent's work.
    // They are real tree changes so they are not hidden, but they must not read
    // the same as a source edit when you are scanning for what broke.
    var files = (o.files||[]).length
      ? (o.files||[]).map(function(f){
          var hk = f.indexOf('.taste/')===0;
          return '<span class="tag'+(hk?' hk':'')+'" title="'
               + (hk?'harness bookkeeping':'agent change')+'">'+E(f)+'</span>';
        }).join('')
      : '<span class="empty">no files changed</span>';
    $('detail').innerHTML =
      '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px">'
      + '<strong class="mono">observation '+E(o.seq)+'</strong>'
      + '<span class="tag">'+E(o.step_id||'—')+' attempt '+E(o.attempt)+'</span>'
      + '<span class="tag">'+E(o.trigger)+'</span>'
      + '<span class="tag">$'+ (o.cost_work_usd||0).toFixed(4) +' work</span>'
      + mon + '</div>'
      + brk
      + '<div style="margin-bottom:8px">'+files+'</div>'
      + (o.diff ? '<pre>'+colourDiff(o.diff)+'</pre>'
                  + (o.diff_truncated?'<p class="empty">diff truncated</p>':'')
                : '<p class="empty">no diff recorded</p>');
  }
  $('obs').addEventListener('click', function(e){
    var b = e.target.closest('button'); if(b) showObs(+b.dataset.i);
  });
  if(obs.length) showObs(0);

  // --- steps
  var st = D.steps||[];
  $('steps').innerHTML = st.length ? (
    '<thead><tr><th>step</th><th>description</th><th>verdict</th><th>what happened</th><th>tools</th></tr></thead><tbody>'
    + st.map(function(s){
        var v = s.passed===true?'<span class="pill pass">passed</span>'
              : (s.passed===false?'<span class="pill fail">failed</span>':'<span class="pill idle">not run</span>');
        var tags = (s.attempts>1?'<span class="tag warn">'+s.attempts+' attempts</span>':'')
                 + (s.rolled_back?'<span class="tag rec">rolled back</span>':'')
                 + (s.recovery||[]).map(function(a){return '<span class="tag rec">'+E(a)+'</span>';}).join('');
        var card = (s.cards||[])[ (s.cards||[]).length-1 ] || {};
        var tools = card.tool_calls!=null
          ? E(card.tool_calls)+(card.tool_errors?' ('+E(card.tool_errors)+' err)':'') : '—';
        return '<tr><td class="mono">'+E(s.id)+'</td><td>'+E(s.description)+'</td><td>'+v
             + '</td><td>'+(tags||'<span class="empty">—</span>')+'</td><td class="mono">'+tools+'</td></tr>';
      }).join('') + '</tbody>'
  ) : '<tbody><tr><td class="empty">No plan recorded.</td></tr></tbody>';

  // --- events, filterable by lane
  var evs = D.events||[], active = null;
  var lanes = []; evs.forEach(function(e){ if(lanes.indexOf(e.lane)<0) lanes.push(e.lane); });
  $('lanes').innerHTML = ['all'].concat(lanes).map(function(l){
    return '<button data-lane="'+E(l)+'" aria-pressed="'+(l==='all')+'">'+E(l)+'</button>';
  }).join('');
  function drawEvents(){
    var rows = evs.filter(function(e){return !active || e.lane===active;});
    $('events').innerHTML = '<thead><tr><th>t+</th><th>lane</th><th>event</th><th>detail</th></tr></thead><tbody>'
      + rows.map(function(e){
          return '<tr><td class="mono">'+e.t.toFixed(2)+'s</td><td class="mono">'+E(e.lane)
               + '</td><td class="mono" style="color:'+E(e.colour)+'">'+E(e.kind)+'</td><td>'+E(e.label)+'</td></tr>';
        }).join('') + '</tbody>';
  }
  $('lanes').addEventListener('click', function(e){
    var b = e.target.closest('button'); if(!b) return;
    active = b.dataset.lane==='all' ? null : b.dataset.lane;
    Array.prototype.forEach.call($('lanes').children, function(x){
      x.setAttribute('aria-pressed', x.dataset.lane===b.dataset.lane ? 'true':'false');
    });
    drawEvents();
  });
  drawEvents();

  $('models').innerHTML = (D.models||[]).length
    ? '<thead><tr><th>role</th><th>model</th><th>temp</th></tr></thead><tbody>'
      + D.models.map(function(m){return '<tr><td class="mono">'+E(m.role)+'</td><td class="mono">'+E(m.model)
        +'</td><td class="mono">'+E(m.temperature==null?'—':m.temperature)+'</td></tr>';}).join('')+'</tbody>'
    : '<tbody><tr><td class="empty">No model manifest.</td></tr></tbody>';

  $('branches').innerHTML = (D.branches||[]).length
    ? '<thead><tr><th>branch</th><th>step</th><th>outcome</th></tr></thead><tbody>'
      + D.branches.map(function(b){return '<tr><td class="mono">'+E(b.name)+'</td><td class="mono">'+E(b.step_id)
        +'</td><td><span class="pill '+(b.outcome==='merged'?'pass':'warn')+'">'+E(b.outcome)+'</span></td></tr>';}).join('')+'</tbody>'
    : '<tbody><tr><td class="empty">Steps ran on the session branch.</td></tr></tbody>';

  $('foot').innerHTML = 'session <span class="mono">'+E(D.session)+'</span> &middot; branch <span class="mono">'
    + E(D.branch)+'</span> &middot; harness <span class="mono">'+E((D.harness_sha||'').slice(0,12))+'</span>';
})();
</script>
"""

_INDEX = """<title>__TITLE__</title>
<style>""" + _CSS + """</style>
<div class="wrap">
  <header><h1>Sweep</h1><p id="sub"></p></header>
  <div class="strip" id="strip"></div>
  <section class="panel">
    <div class="row" id="filters"></div>
    <div class="scroll"><table id="t"></table></div>
  </section>
  <p class="foot">Click a row's report link to open that cell. Generated by <span class="mono">taste.viz</span>.</p>
</div>
<script id="payload" type="application/json">__DATA__</script>
<script>
(function(){
  var D = JSON.parse(document.getElementById('payload').textContent);
  var rows = D.rows||[];
  var E = function(s){var d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;};
  var $ = function(i){return document.getElementById(i);};

  var arms = []; rows.forEach(function(r){ if(arms.indexOf(r.arm)<0) arms.push(r.arm); });
  var totalCost = rows.reduce(function(a,r){return a+(r.billed_usd||0);},0);
  var totalSilent = rows.reduce(function(a,r){return a+(r.silent||0);},0);
  var totalEp = rows.reduce(function(a,r){return a+(r.episodes||0);},0);
  var failed = rows.filter(function(r){return r.error||r.status==='error';}).length;

  $('sub').textContent = rows.length + ' cells across ' + arms.length + ' arm(s)';
  var cells = [['cells',rows.length,''],['arms',arms.length,''],
    ['regressions',totalEp,''],['silent',totalSilent,totalSilent>0?'alert':''],
    ['errored',failed,failed>0?'alert':''],['spend','$'+totalCost.toFixed(2),'']];
  $('strip').innerHTML = cells.map(function(c){
    return '<div class="cell '+c[2]+'"><div class="lab">'+E(c[0])+'</div><div class="val">'+E(c[1])+'</div></div>';}).join('');

  var active=null;
  $('filters').innerHTML = ['all'].concat(arms).map(function(a){
    return '<button data-arm="'+E(a)+'" aria-pressed="'+(a==='all')+'">'+E(a)+'</button>';}).join('');
  function draw(){
    var rs = rows.filter(function(r){return !active||r.arm===active;});
    $('t').innerHTML = '<thead><tr><th>instance</th><th>arm</th><th>t</th><th>status</th><th>steps</th>'
      + '<th>obs</th><th>regr</th><th>silent</th><th>cost</th><th>report</th></tr></thead><tbody>'
      + (rs.length ? rs.map(function(r){
          var cls = r.status==='completed'?'pass':(r.status==='error'?'fail':'warn');
          return '<tr><td class="mono">'+E(r.task)+'</td><td class="mono">'+E(r.arm)+'</td><td class="mono">'+E(r.trial)
            +'</td><td><span class="pill '+cls+'">'+E(r.status)+'</span></td><td class="mono">'+E(r.steps)
            +'</td><td class="mono">'+E(r.observations)+'</td><td class="mono">'+E(r.episodes)
            +'</td><td class="mono" style="color:'+(r.silent?'var(--fail)':'inherit')+'">'+E(r.silent)
            +'</td><td class="mono">$'+(r.billed_usd||0).toFixed(4)+'</td><td>'
            +(r.report?'<a href="'+E(r.report)+'">open</a>':'<span class="empty">—</span>')+'</td></tr>';
        }).join('') : '<tr><td class="empty" colspan="10">No cells.</td></tr>') + '</tbody>';
  }
  $('filters').addEventListener('click', function(e){
    var b=e.target.closest('button'); if(!b) return;
    active = b.dataset.arm==='all'?null:b.dataset.arm;
    Array.prototype.forEach.call($('filters').children,function(x){
      x.setAttribute('aria-pressed', x.dataset.arm===b.dataset.arm?'true':'false');});
    draw();
  });
  draw();
})();
</script>
"""
