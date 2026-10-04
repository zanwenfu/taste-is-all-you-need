"""Replay one recorded decision of a Taste run with the code on PYTHONPATH.

    python scripts/replay_decision.py coordinator WORKSPACE [--generation N|last] [--calls 3]
    python scripts/replay_decision.py monitor WORKSPACE --branch SUFFIX [--calls 3]
    python scripts/replay_decision.py certifier WORKSPACE --branch SUFFIX [--calls 3]

WORKSPACE is the git repository a run's memory lives in (for a benchmark
trial, its agent-state/workspace). It is copied to a temporary directory and
the copy is opened; the original is never touched.

coordinator: rebuilds the coordinator's prompt for one planning request (the
last one before the closing reply, or generation N) and prints its SHA-256
beside the one recorded when the request was sent. Run with the code that
made the decision, the two match; that is the check that the replay is
faithful. Run with changed code, the prompt is what that code would have
sent in the same situation.

monitor: rebuilds the step judgement that stopped a worker's run (the first
judged wrong or lost) from that run's monitor state: the batch as judged, the
memory state it saw, and the events before it.

certifier: rebuilds a worker run's last certification from its monitor state:
the State it judged, the context it was given and the earlier findings.

With --calls N, the same model is asked N times with the call settings the run
used (Azure credentials from AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_KEY),
and what each answer decided is printed. Each call is paid.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from taste.brains.azure_execution_policy import POLICY_KEY, AzureExecutionPolicy
from taste.brains.central_planner import PLANNER_SYSTEM, CentralPlanner, PlanningRequest
from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.monitor import MonitorBrain
from taste.brains.records import Assignment
from taste.memstore import Store

_STOPS = ("wrong", "lost")


def copy_workspace(source: Path, into: Path | None = None) -> Path:
    """A private copy of a run's memory, its worktree registrations left out.

    The copy is ``workspace`` in a directory of its own, where the store also
    opens its worktrees; removing that directory removes both.
    """
    parent = Path(tempfile.mkdtemp(prefix="taste-replay-")) if into is None else into
    target = parent / "workspace"
    shutil.copytree(source, target, symlinks=True,
                    ignore=lambda folder, _names: ["worktrees"] if folder.endswith(".git") else [])
    return target


def _branches(workspace: Path) -> list[str]:
    refs = subprocess.run(["git", "-C", str(workspace), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
                          capture_output=True, text=True, check=True).stdout.split()
    return [ref.rsplit("/", 1)[-1] for ref in refs]


class _RenderOnly:
    def complete(self, **_):
        raise RuntimeError("a replay renders prompts; it sends nothing through the run's transport")


def coordinator_prompt(store: Store, generation: str = "last"):
    """The planning request, the prompt the code on the path builds for it, and the recorded hashes."""
    head = store.view("central-control").head
    requests = [PlanningRequest.from_json(head.read(path)) for path in head.files()
                if path.startswith(".taste/planner/operations/") and path.endswith("/request.json")]
    candidates = [item for item in requests if "closing" not in item.operation_id]
    wanted = max(item.generation for item in candidates) if generation == "last" else int(generation)
    # A generation can have several requests (a refused plan is asked for
    # again); the one to replay is the one whose plan was accepted.
    accepted = {json.loads(head.read(path)).get("metadata", {}).get("request_id")
                for path in head.files() if path.startswith(".taste/planner/plans/")}
    matching = [item for item in candidates if item.generation == wanted]
    request = next((item for item in matching if item.request_id in accepted), matching[0])
    sent = [json.loads(head.read(path)) for path in head.files()
            if path.startswith(".taste/planner/transport/") and path.endswith("/intent.json")]
    recorded = [item["binding"]["prompt_sha256"] for item in sent
                if item["scope"]["planning_request_id"] == request.request_id]
    raw_policy = request.goal.metadata.get(POLICY_KEY)
    policy = None if raw_policy is None else AzureExecutionPolicy.from_dict(raw_policy)
    prompt = CentralPlanner(store, transport=_RenderOnly(), azure_policy=policy)._prompt(request)
    return request, prompt, recorded, policy


def _llm(azure, run_id):
    from taste.llm import LLM

    return LLM(azure_openai=azure, budget_usd=1.0, cap_on="billed", max_attempts=1, load_env_file=False,
               run_id=run_id)


def ask_coordinator(prompt: str, policy: AzureExecutionPolicy, calls: int, environ) -> list[dict]:
    """What the coordinator's model decides from this prompt, ``calls`` times, with the run's settings."""
    llm = _llm(policy.azure_config(environ), "replay-coordinator")
    answers = []
    for _ in range(calls):
        completion = llm.call(model=policy.planner_model, system=PLANNER_SYSTEM,
                              messages=[{"role": "user", "content": prompt}], tools=None,
                              max_tokens=policy.planner_max_output_tokens, temperature=0.0, role="planner",
                              json_output="json" in prompt.lower(),
                              **({"effort": policy.planner_effort} if policy.planner_effort else {}))
        text = "".join(completion.text_blocks)
        try:
            answers.append(json.loads(text))
        except ValueError:
            answers.append({"not_json": text[:400]})
    return answers


class _ReplayView:
    """What the monitor saw: the memory state it judged against, and the run before the batch."""

    def __init__(self, head, events, start):
        self.head = head
        self.observed_events = tuple(events)
        self.earlier_events = tuple(events[:start])
        self.worker_running = True

    def exists(self):
        return True


def monitor_judgement(store: Store, workspace: Path, session: str, suffix: str):
    """The stopping judgement of one worker's run, rebuilt: contract, batch, view, and the record."""
    name = next(branch for branch in _branches(workspace) if branch.endswith(suffix))
    live = store.view(name)
    head = live.head
    raw_assignment = head.read("assignment.json")
    assignment = None if raw_assignment is None else Assignment.from_dict(json.loads(raw_assignment))
    contract = assignment.contract if assignment is not None else Contract.from_dict(
        json.loads(head.read(CONTRACT_PATH)))
    [sidecar] = glob.glob(str(workspace / ".git" / "memstore-sidecars" / "v2" / session / name / "monitor.*"))
    state = json.loads(Path(sidecar).read_text())
    action = next(item for item in state["actions"] if item["judgement"]["severity"] in _STOPS)
    batch = action["batch"]
    # As the monitor read them: the committed turns and the journal extending them.
    events = list(MonitorBrain._judgeable(*head.transcript.turns, *live.pending_turns()))
    start = next(index for index in range(len(events)) if events[index:index + len(batch)] == batch)
    view = _ReplayView(store.state(action["observed_head"]), events, start)
    return name, assignment, contract, batch, view, action


def _monitor_judge(assignment: Assignment, environ):
    """The run's monitor judge: its model, output limit, effort and view sizes."""
    from taste.brains.azure_worker_policy import AzureWorkerPolicy
    from taste.brains.monitor_judge import LLMMonitorJudge

    try:
        from taste.brains.responses_monitor import size_views
    except ImportError:  # code from before the rule had a name: the same rule
        from taste.brains import monitor_judge as m

        def size_views(judge, max_request_bytes):
            room = max_request_bytes - m.MAX_MONITOR_PROMPT_BYTES
            judge.transcript_view_bytes = max(m.TRANSCRIPT_VIEW_BYTES, min(m.MAX_TRANSCRIPT_VIEW_BYTES, room // 2))
            judge.artifact_view_bytes = max(m.MAX_INLINE_ARTIFACT_BYTES, min(m.MAX_ARTIFACT_VIEW_BYTES, room // 8))
            judge.max_prompt_bytes = max(m.MAX_MONITOR_PROMPT_BYTES, max_request_bytes * 3 // 4)
            return judge

    policy = AzureWorkerPolicy.from_assignment(assignment)
    llm = _llm(policy.azure_config(environ), "replay-monitor")

    class Calls:
        def call(self, *, model, system, messages, tools, max_tokens, temperature, role):
            return llm.call(model=model, system=system, messages=messages, tools=None, max_tokens=max_tokens,
                            temperature=0.0, role="monitor", effort="low")

    judge = LLMMonitorJudge(Calls(), model=policy.monitor.model, max_tokens=policy.monitor.max_output_tokens,
                            json_prefill=False)
    return size_views(judge, policy.monitor.max_request_bytes)


def ask_monitor(assignment: Assignment, contract: Contract, batch, view, calls: int, environ) -> list:
    """What the monitor's model judges of this batch, ``calls`` times, with the run's settings."""
    judge = _monitor_judge(assignment, environ)
    return [judge(contract, list(batch), view) for _ in range(calls)]


def certifier_judgement(store: Store, workspace: Path, suffix: str):
    """The last certification of one worker's run, rebuilt: the State, context and findings it judged."""
    name = next(branch for branch in _branches(workspace) if branch.endswith(suffix))
    head = store.view(name).head
    raw_assignment = head.read("assignment.json")
    assignment = None if raw_assignment is None else Assignment.from_dict(json.loads(raw_assignment))
    contract = assignment.contract if assignment is not None else Contract.from_dict(
        json.loads(head.read(CONTRACT_PATH)))
    raw_report = head.read("worker-report.json")
    run_id = None if raw_report is None else json.loads(raw_report).get("run_id")
    monitor = MonitorBrain(store, contract, None, run_id=run_id)
    assessment = monitor.state.terminal_assessments[-1]
    findings = list(monitor._terminal_findings())
    if tuple(item["id"] for item in findings) != assessment.finding_ids:
        raise RuntimeError("the monitor's findings differ from those the certification judged")
    return (name, assignment, contract, store.state(assessment.state_id),
            json.loads(assessment.context_json), findings, assessment)


def ask_certifier(assignment: Assignment, contract: Contract, state, context, findings, calls: int, environ):
    """What the certifier's model decides of this State, ``calls`` times, with the run's settings."""
    judge = _monitor_judge(assignment, environ)
    return [judge.judge_terminal(contract, state, context, list(findings)) for _ in range(calls)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("role", choices=("coordinator", "monitor", "certifier"))
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--session", default="terminal-trial", help="the memory session (default: a trial's)")
    parser.add_argument("--generation", default="last", help="coordinator: the plan's generation, or last")
    parser.add_argument("--branch", help="monitor: the end of the worker branch's name")
    parser.add_argument("--calls", type=int, default=0, help="paid model calls to make (default: none)")
    arguments = parser.parse_args(argv)
    workspace = copy_workspace(arguments.workspace)
    store = None
    try:
        store = Store.open(workspace, arguments.session)
        if arguments.role == "coordinator":
            request, prompt, recorded, policy = coordinator_prompt(store, arguments.generation)
            print(f"generation {request.generation}, operation {request.operation_id}")
            print("rebuilt  ", hashlib.sha256(prompt.encode()).hexdigest())
            print("recorded ", " ".join(recorded) or "(none)")
            if arguments.calls:
                for index, plan in enumerate(ask_coordinator(prompt, policy, arguments.calls, os.environ)):
                    verdicts = [item.get("verdict") for item in plan.get("assessment") or []]
                    print(f"--- {index}: complete={plan.get('complete')} verdicts={verdicts} "
                          f"assignments={len(plan.get('assignments') or [])}")
                    print("    rationale:", str(plan.get("rationale"))[:400])
        elif arguments.role == "certifier":
            if not arguments.branch:
                parser.error("certifier needs --branch")
            name, assignment, contract, state, context, findings, assessment = certifier_judgement(
                store, workspace, arguments.branch)
            print(f"branch {name}: State {state.id}, {len(findings)} earlier findings")
            print("recorded:", assessment.judgement.severity.value, "acceptable" if assessment.acceptable
                  else "refused", "|", assessment.judgement.reason[:300])
            if arguments.calls:
                for index, decision in enumerate(ask_certifier(assignment, contract, state, context, findings,
                                                               arguments.calls, os.environ)):
                    print(f"--- {index}: {decision.judgement.severity.value} | {decision.judgement.reason[:300]}")
        else:
            if not arguments.branch:
                parser.error("monitor needs --branch")
            name, assignment, contract, batch, view, action = monitor_judgement(
                store, workspace, arguments.session, arguments.branch)
            print(f"branch {name}: batch of {len(batch)} after {len(view.earlier_events)} earlier events")
            print("recorded:", action["judgement"]["severity"], "|", action["judgement"]["reason"][:300])
            if arguments.calls:
                for index, judgement in enumerate(ask_monitor(assignment, contract, batch, view,
                                                              arguments.calls, os.environ)):
                    print(f"--- {index}: {judgement.severity.value} | {judgement.reason[:300]}")
    finally:
        if store is not None:
            store.close()
        # The copy and the worktrees opened beside it.
        shutil.rmtree(workspace.parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
