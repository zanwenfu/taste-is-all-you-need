"""A live console for runs in progress, not a report you generate afterwards.

:mod:`taste.viz` renders a finished run. That is the wrong shape while a run is
still going: an agent working through a real repository takes minutes per step,
and the question during those minutes is *what is it doing right now* — which
core is active, which attempt, what the Monitor just said, how much of the
budget is gone.

So this serves the same data over HTTP and streams new events as they are
written. Deliberately built on the standard library alone: no framework, no
build step, no new dependency. A research harness that needs npm to show you
its own state is a harness people stop looking at.

Two endpoints do the work. ``/api/runs`` scans a directory for anything that
looks like a run, and ``/api/stream/<id>`` tails that run's event log as
server-sent events. SSE rather than websockets because tailing a file is
one-directional and SSE reconnects by itself.

**A note on what "live" can and cannot mean here.** The event log is the
harness's own record, written as it goes, so the feed is genuinely live. The
regression timeline is not: probes are replayed *after* a run, against the
pinned image, so a run in flight shows observations without verdicts. The UI
says so rather than leaving an empty panel that reads as "nothing broke".
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

POLL_SECONDS = 0.5
"""How often a stream checks its log for new lines. Cheap: one stat call."""


@dataclass
class RunHandle:
    """One run discovered on disk, and where its pieces live."""

    run_id: str
    workspace: Path
    events: Path
    instance: str = ""
    arm: str = ""
    trial: int = 0
    status: str = "unknown"
    evidence: Path | None = None
    started: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "instance": self.instance or self.workspace.name,
            "arm": self.arm,
            "trial": self.trial,
            "status": self.status,
            "workspace": str(self.workspace),
            "started": self.started,
            "events": (self.events.exists() and self.events.stat().st_size) or 0,
        }


@dataclass
class Discovery:
    """Everything under a root that looks like a run.

    A workspace is recognised by its event log rather than by a manifest,
    because the log is written from the first moment of a run — a run that
    crashed during planning still has one, and that is exactly the run someone
    opens the console to look at.
    """

    root: Path
    runs: dict[str, RunHandle] = field(default_factory=dict)

    def scan(self) -> list[RunHandle]:
        found: dict[str, RunHandle] = {}
        for events in sorted(self.root.rglob(".git/taste/events.jsonl")):
            workspace = events.parent.parent.parent
            run_id = str(workspace.relative_to(self.root)).replace("/", "~")
            handle = RunHandle(
                run_id=run_id,
                workspace=workspace,
                events=events,
                started=events.stat().st_mtime,
            )
            _enrich(handle, self.root)
            found[run_id] = handle
        self.runs = found
        return sorted(found.values(), key=lambda h: -h.started)

    def get(self, run_id: str) -> RunHandle | None:
        if run_id not in self.runs:
            self.scan()
        return self.runs.get(run_id)


def _enrich(handle: RunHandle, root: Path) -> None:
    """Fill in what the ledger knows, when a sweep wrote one."""
    parts = handle.workspace.relative_to(root).parts
    if len(parts) >= 3:
        handle.instance, handle.arm = parts[-3], parts[-2]
        trial = parts[-1].lstrip("t")
        handle.trial = int(trial) if trial.isdigit() else 0

    for ledger in root.rglob(f"{handle.instance}__{handle.arm}__t{handle.trial}.json"):
        try:
            raw = json.loads(ledger.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "evidence" in ledger.parts:
            handle.evidence = ledger
        else:
            handle.status = str(raw.get("status", handle.status))
    if handle.status == "unknown":
        handle.status = _status_from_events(handle.events)


def _status_from_events(events: Path) -> str:
    """A run with no ledger row has not finished; read its own last word."""
    try:
        lines = events.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unknown"
    for line in reversed(lines[-40:]):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") in ("run.done", "run.halt"):
            return str((event.get("payload") or {}).get("status", "finished"))
    return "running" if lines else "starting"


def read_events(path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Events from a byte offset, plus the new offset.

    Reads by byte position rather than line count so a partially written final
    line is left alone until it is complete — otherwise a fast run streams
    truncated JSON to the browser.
    """
    if not path.exists():
        return [], offset
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        chunk = handle.read()
        end = handle.tell()
    lines = chunk.split("\n")
    if not chunk.endswith("\n") and lines:
        incomplete = lines.pop()
        end -= len(incomplete.encode("utf-8"))
    out: list[dict[str, Any]] = []
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out, end


def run_payload(handle: RunHandle) -> dict[str, Any]:
    """The full picture for one run, reusing the report builder.

    Diffs are skipped: the console is for watching, and a live run can be
    re-read many times a minute. The finished report is where diffs belong.
    """
    from taste.viz import build_payload

    evidence: dict[str, Any] = {}
    if handle.evidence and handle.evidence.exists():
        try:
            evidence = json.loads(handle.evidence.read_text())
        except (OSError, json.JSONDecodeError):
            evidence = {}
    payload = build_payload(
        handle.workspace, evidence=evidence, arm=handle.arm,
        instance=handle.instance, with_diffs=False,
    )
    data = asdict(payload)
    data["status"] = handle.status
    data["live"] = handle.status in ("running", "starting")
    if data["live"]:
        data.setdefault("notes", []).append(
            "This run is still going. Probes are replayed after a run finishes, "
            "so regressions are not yet known — not known to be absent."
        )
    return data


# ---------------------------------------------------------------- http


class _Handler(BaseHTTPRequestHandler):
    discovery: Discovery  # bound per-server below
    server_version = "taste"

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # a research console should not spam the terminal it runs in

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        try:
            if route in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", CONSOLE.encode("utf-8"))
            elif route == "/api/runs":
                self._json([h.summary() for h in self.discovery.scan()])
            elif route.startswith("/api/run/"):
                self._run(unquote(route[len("/api/run/"):]))
            elif route.startswith("/api/stream/"):
                self._stream(unquote(route[len("/api/stream/"):]))
            else:
                self._send(404, "text/plain", b"not found")
        except BrokenPipeError:
            pass  # the browser navigated away mid-response; not an error
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _run(self, run_id: str) -> None:
        handle = self.discovery.get(run_id)
        if handle is None:
            self._json({"error": "no such run"}, status=404)
            return
        self._json(run_payload(handle))

    def _stream(self, run_id: str) -> None:
        """Tail one run's events as server-sent events.

        The client sends its byte offset so a reconnect resumes rather than
        replaying the run from the beginning.
        """
        handle = self.discovery.get(run_id)
        if handle is None:
            self._json({"error": "no such run"}, status=404)
            return
        query = urlparse(self.path).query
        offset = 0
        for part in query.split("&"):
            if part.startswith("offset="):
                with_digits = part[len("offset="):]
                offset = int(with_digits) if with_digits.isdigit() else 0

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        idle = 0.0
        while True:
            events, offset = read_events(handle.events, offset)
            if events:
                idle = 0.0
                for event in events:
                    self._sse("event", {"offset": offset, "event": event})
            else:
                idle += POLL_SECONDS
                if idle >= 15:
                    self._sse("ping", {"offset": offset})  # keep proxies awake
                    idle = 0.0
            status = _status_from_events(handle.events)
            if status not in ("running", "starting"):
                self._sse("done", {"status": status, "offset": offset})
                return
            time.sleep(POLL_SECONDS)

    def _sse(self, name: str, data: dict[str, Any]) -> None:
        body = f"event: {name}\ndata: {json.dumps(data)}\n\n"
        self.wfile.write(body.encode("utf-8"))
        self.wfile.flush()

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(root: Path, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Start the console over ``root``.

    Bound to localhost by default and on purpose. The console serves file
    contents and command output from a machine running an agent with API
    credentials; it has no authentication and is not built to have any.
    """
    mimetypes.init()
    discovery = Discovery(root=Path(root).resolve())
    discovery.scan()
    handler = type("_Bound", (_Handler,), {"discovery": discovery})
    return ThreadingHTTPServer((host, port), handler)


def serve_forever(root: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = serve(root, host=host, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        httpd.shutdown()


# ---------------------------------------------------------------- console


CONSOLE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>taste console</title>
<style>
:root{--ground:#F5F7F6;--panel:#FFF;--sunk:#EDF1F0;--ink:#0F1619;--dim:#5A686E;
--rule:#DDE4E3;--sig:#1F6F6C;--pass:#2C8B52;--fail:#C43D35;--warn:#B37B12;--rec:#6A4BE0;}
@media (prefers-color-scheme:dark){:root{--ground:#0B1013;--panel:#131B20;--sunk:#0F1619;
--ink:#DCE6E9;--dim:#7C8F97;--rule:#222F35;--sig:#4FB3AE;--pass:#3FA86A;--fail:#E05A50;
--warn:#D2971F;--rec:#8E72FF;}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
display:grid;grid-template-columns:290px 1fr;height:100vh;overflow:hidden}
.mono,code,pre,th,.tag,.pill{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
aside{border-right:1px solid var(--rule);background:var(--panel);overflow-y:auto;padding:14px}
main{overflow-y:auto;padding:20px 24px 60px}
h1{font:600 13px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase;
color:var(--sig);margin:0 0 14px}
h2{font:600 11px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.11em;text-transform:uppercase;
color:var(--dim);margin:0 0 9px}
.run{padding:9px 11px;border:1px solid var(--rule);border-radius:8px;margin-bottom:7px;cursor:pointer;background:var(--panel)}
.run:hover{border-color:var(--sig)}
.run[aria-selected="true"]{border-color:var(--sig);background:var(--sunk)}
.run .id{font:600 12px/1.35 ui-monospace,Menlo,monospace;word-break:break-all}
.run .meta{font-size:11px;color:var(--dim);margin-top:3px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;flex:none}
.dot.running{background:var(--sig);animation:pulse 1.4s ease-in-out infinite}
.dot.completed{background:var(--pass)}.dot.failed,.dot.error{background:var(--fail)}
.dot.budget{background:var(--warn)}.dot.unknown{background:var(--dim)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
@media (prefers-reduced-motion:reduce){.dot.running{animation:none}}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(104px,1fr));gap:1px;
background:var(--rule);border:1px solid var(--rule);border-radius:9px;overflow:hidden;margin-bottom:18px}
.cell{background:var(--panel);padding:10px 12px}
.cell .lab{font:10px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
.cell .val{font:600 19px/1.2 ui-monospace,Menlo,monospace;margin-top:3px;font-variant-numeric:tabular-nums}
.cell.alert .val{color:var(--fail)}.cell.good .val{color:var(--pass)}.cell.live .val{color:var(--sig)}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:9px;padding:15px 17px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{text-align:left;font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);
padding:0 10px 6px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:7px 10px 7px 0;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:10px;font-weight:600;border:1px solid currentColor}
.pill.pass{color:var(--pass)}.pill.fail{color:var(--fail)}.pill.idle{color:var(--dim)}.pill.warn{color:var(--warn)}
.tag{font-size:10px;padding:1px 6px;border-radius:4px;background:var(--sunk);color:var(--dim);margin-right:4px}
.tag.rec{color:var(--rec)}.tag.warn{color:var(--warn)}
.feed{max-height:340px;overflow-y:auto;font-family:ui-monospace,Menlo,monospace;font-size:11.5px}
.ev{display:flex;gap:9px;padding:3px 0;border-bottom:1px solid var(--rule)}
.ev:last-child{border-bottom:none}
.ev .t{color:var(--dim);flex:none;width:52px;text-align:right;font-variant-numeric:tabular-nums}
.ev .k{flex:none;width:150px;font-weight:600}
.ev .d{color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lane{display:inline-block;font-size:10px;padding:2px 7px;border-radius:4px;background:var(--sunk);color:var(--sig);margin-right:5px}
.note{background:var(--sunk);border-left:3px solid var(--warn);padding:8px 12px;border-radius:0 6px 6px 0;
font-size:12px;margin-bottom:9px}
.empty{color:var(--dim);font-style:italic;font-size:12.5px}
.bar{height:5px;background:var(--sunk);border-radius:3px;overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%;background:var(--sig)}
.bar i.over{background:var(--fail)}
</style></head><body>
<aside><h1>runs</h1><div id="runs"></div></aside>
<main id="main"><p class="empty">Select a run.</p></main>
<script>
const E=s=>{const d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;};
const $=i=>document.getElementById(i);
let current=null, stream=null, offset=0, seen=[];

const LANES={run:'kernel',wave:'kernel',journal:'kernel',shadow:'kernel',plan:'planner',
  step:'worker',worker:'worker',monitor:'monitor',recovery:'recovery',guard:'guard',
  merge:'merge',worktree:'merge'};
const COLOR=k=>k==='monitor.verdict'?'':k.startsWith('recovery')?'var(--rec)':
  k.startsWith('guard')?'var(--warn)':/halt|error|abort/.test(k)?'var(--fail)':
  /done/.test(k)?'var(--pass)':'var(--sig)';

async function loadRuns(){
  const rs=await (await fetch('/api/runs')).json();
  $('runs').innerHTML = rs.length ? rs.map(r=>`
    <div class="run" data-id="${E(r.run_id)}" aria-selected="${r.run_id===current}">
      <div class="id">${E(r.instance)}</div>
      <div class="meta"><span class="dot ${E(r.status)}"></span>${E(r.status)}
        ${r.arm?`<span class="tag">${E(r.arm)}</span>`:''}
        ${r.trial?`<span class="tag">t${E(r.trial)}</span>`:''}</div>
    </div>`).join('') : '<p class="empty">No runs found under this root.</p>';
}

async function openRun(id){
  current=id; offset=0; seen=[];
  if(stream){stream.close(); stream=null;}
  await loadRuns();
  const d=await (await fetch('/api/run/'+encodeURIComponent(id))).json();
  if(d.error){$('main').innerHTML=`<p class="empty">${E(d.error)}</p>`;return;}
  render(d);
  stream=new EventSource('/api/stream/'+encodeURIComponent(id)+'?offset=0');
  stream.addEventListener('event',e=>{const m=JSON.parse(e.data);offset=m.offset;push(m.event);});
  stream.addEventListener('done',async()=>{stream.close();stream=null;
    const fresh=await (await fetch('/api/run/'+encodeURIComponent(id))).json();
    render(fresh); loadRuns();});
}

function render(d){
  const silent=(d.episodes||[]).filter(e=>e.detected_seq_attributed==null).length;
  const cells=[['status',d.status,d.live?'live':(d.status==='completed'?'good':'')],
    ['arm',d.arm||'—',''],['steps',(d.steps||[]).length,''],
    ['observations',(d.observations||[]).length,''],
    ['regressions',d.live?'—':(d.episodes||[]).length,''],
    ['silent',d.live?'—':silent,silent>0?'alert':''],
    ['cost','$'+(d.cost_usd||0).toFixed(4),''],
    ['elapsed',(d.elapsed_s||0).toFixed(0)+'s','']];
  $('main').innerHTML=`
    <h2>${E(d.instance||d.session)}</h2>
    <p class="empty" style="margin-bottom:14px">${E((d.task||'').split('\\n')[0].slice(0,150))}</p>
    <div class="strip">${cells.map(c=>`<div class="cell ${c[2]}"><div class="lab">${E(c[0])}</div><div class="val">${E(c[1])}</div></div>`).join('')}</div>
    <div id="notes">${(d.notes||[]).map(n=>`<div class="note">${E(n)}</div>`).join('')}</div>
    <div class="panel"><h2>Steps — how each one went</h2><div id="steps"></div></div>
    <div class="panel"><h2>Live activity</h2><div id="lanes"></div><div class="feed" id="feed"></div></div>
    <div class="panel"><h2>Models</h2><div id="models"></div></div>`;
  drawSteps(d.steps||[]); drawModels(d.models||[]);
  seen=(d.events||[]).map(e=>({t:e.t,kind:e.kind,lane:e.lane,label:e.label}));
  drawFeed();
}

function drawSteps(steps){
  $('steps').innerHTML = steps.length ? `<table><thead><tr><th>step</th><th>description</th>
    <th>verdict</th><th>attempts</th><th>tools</th><th>what happened</th></tr></thead><tbody>`
    + steps.map(s=>{
      const v=s.passed===true?'<span class="pill pass">passed</span>':
        s.passed===false?'<span class="pill fail">failed</span>':'<span class="pill idle">running</span>';
      const card=(s.cards||[])[(s.cards||[]).length-1]||{};
      const tools=card.tool_calls!=null?E(card.tool_calls)+(card.tool_errors?` <span class="tag warn">${E(card.tool_errors)} err</span>`:''):'—';
      const tags=(s.rolled_back?'<span class="tag rec">rolled back</span>':'')
        +(s.recovery||[]).map(a=>`<span class="tag rec">${E(a)}</span>`).join('');
      return `<tr><td class="mono">${E(s.id)}</td><td>${E(s.description)}</td><td>${v}</td>
        <td class="mono">${E(s.attempts||0)}</td><td class="mono">${tools}</td>
        <td>${tags||'<span class="empty">—</span>'}</td></tr>`;}).join('')+'</tbody></table>'
    : '<p class="empty">No plan yet — the planner has not returned.</p>';
}

function drawModels(models){
  $('models').innerHTML = models.length
    ? `<table><thead><tr><th>role</th><th>model</th><th>temp</th></tr></thead><tbody>`
      + models.map(m=>`<tr><td class="mono">${E(m.role)}</td><td class="mono">${E(m.model)}</td>
        <td class="mono">${E(m.temperature==null?'—':m.temperature)}</td></tr>`).join('')+'</tbody></table>'
    : '<p class="empty">No manifest yet.</p>';
}

function push(ev){
  const fam=(ev.kind||'').split('.')[0];
  seen.push({t:null,kind:ev.kind,lane:LANES[fam]||'kernel',
    label:(ev.payload&&(ev.payload.id||ev.payload.action||ev.payload.status))||''});
  drawFeed();
  if(/^(monitor|step|recovery|plan)\\./.test(ev.kind)) refreshSteps();
}

let pending=null;
function refreshSteps(){
  clearTimeout(pending);
  pending=setTimeout(async()=>{
    if(!current) return;
    const d=await (await fetch('/api/run/'+encodeURIComponent(current))).json();
    if(!d.error){drawSteps(d.steps||[]);
      const cells=document.querySelectorAll('.cell .val');
      if(cells[3])cells[3].textContent=(d.observations||[]).length;
      if(cells[6])cells[6].textContent='$'+(d.cost_usd||0).toFixed(4);}
  },600);
}

function drawFeed(){
  const lanes=[...new Set(seen.map(e=>e.lane))];
  $('lanes').innerHTML=lanes.map(l=>`<span class="lane">${E(l)}</span>`).join('');
  const rows=seen.slice(-200).reverse();
  $('feed').innerHTML=rows.map(e=>`<div class="ev">
    <span class="t">${e.t!=null?E(e.t.toFixed(1))+'s':''}</span>
    <span class="k" style="color:${COLOR(e.kind)}">${E(e.kind)}</span>
    <span class="d">${E(e.label)}</span></div>`).join('');
}

document.getElementById('runs').addEventListener('click',e=>{
  const el=e.target.closest('.run'); if(el) openRun(el.dataset.id);});
loadRuns(); setInterval(()=>{if(!stream) loadRuns();},4000);
</script></body></html>
"""
