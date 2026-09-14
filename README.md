# taste is all you need

> **The harness, not the model, is where agents get their taste.**
> An Agent OS with git as the memory substrate: a durable memory layer, a
> central brain that plans and spawns real worker processes, and a monitor
> that observes without touching what it watches.

Full thesis: [Beyond the Harness: An Operating System for AI Agents](https://zanwenfu.com/blog/agent_harness_blog).

---

## The bet, in three bullets

1. **Git *is* the memory system.** Branches are execution contexts, commits are checkpoints, `git show` is demand paging, merge conflicts are coordination signals. What other harnesses bolt on as sidecars — `progress.md`, compaction summaries, retry loops — is already a primitive here.
2. **Multi-core beats single-thread.** A planner decomposes, workers execute in their own OS processes, a monitor judges — **each at a different model size**, each on its own commit boundary. Agents cannot grade their own exams; structural separation fixes that.
3. **Build to delete.** Every component is a bet against the model. When the bet expires, you turn the subsystem off. The OS stays; the knobs change.

## Two layers, one substrate

```
┌──────────────────────────────────────────────────────────────────────┐
│  brains/         central brain: plan → spawn → certify → integrate   │
│                  one OS process per worker, one branch per worker     │
├──────────────────────────────────────────────────────────────────────┤
│  memstore/       durable memory: states, transcripts, verdicts,       │
│                  leases, typed merges — all in git objects and notes  │
└──────────────────────────────────────────────────────────────────────┘
                  kernel/ (taste run)  — the single-process line, CLI-driven
```

`memstore` imports with no model SDK at all; `brains` is an optional extra.
A test holds that boundary ([`tests/test_package_isolation.py`](tests/test_package_isolation.py)).

## What a central run does

A goal goes in. The planner writes a plan, the supervisor spawns a real
subprocess per assignment, each worker gets its own git worktree and branch,
a monitor judges as it works, a terminal certifier reads the finished state,
and certified products are projected into an integration branch.

```python
from taste.brains.central_host import compose_central_runtime
from taste.brains.central_planner import Goal

goal = Goal(
    goal_id="adder",
    task="Create adder.py at the repository root containing def add(a, b): return a + b",
    success_criteria=("adder.py exists and defines add(a, b) returning a + b",),
)

host = compose_central_runtime(repo_root, "session-1", goal)
try:
    while not (outcome := host.cycle()).complete:
        ...          # each call is one crash-replayable reconciliation cycle
finally:
    host.close()
```

Measured on this machine, `claude-opus-4-7` planning and `claude-sonnet-4-6`
working:

| Goal | Workers | Wall clock | Outcome |
|---|---|---|---|
| one file | 1 | 60 s, 63 s | `complete=True`, artifact integrated |
| two files, second depends on the first | 2 | 132 s | both integrated; the test file was written against the delivered implementation |

The planner's own completion reason on the single-file run, quoted from the
durable record:

> *"The integration branch already contains adder.py (artifact-adder-py-001, blob 9bc8bf7b…)"*

That sentence is the whole design in miniature: the planner does not trust a
worker's claim, it reads the integration branch.

## Recorded runs

Every claim above is reproducible; these are the transcripts behind the
single-process line, kept because a harness arguing from measurement should
link its measurements.

- [todo_api/runs/polished.md](examples/todo_api/runs/polished.md) — real Claude, $0.0964, 43 s, 15/15 tests green, one commit per step
- [parallel_demo/runs/parallel.md](examples/parallel_demo/runs/parallel.md) — three workers on real `git worktree`s, 21.5 s vs ~32 s sequential, diamond merges back into the session branch
- [refactor_demo/](examples/refactor_demo/README.md) — the hermetic step-87 rollback, no API key required

`taste dashboard --workspace <ws>` rolls any run's JSONL event stream into a
self-contained HTML artifact ([example](examples/refactor_demo/runs/dashboard.html)).
The event log lives under `.git/` on purpose: inside the tracked tree a
`reset --hard` on rollback would destroy the trace of the moment most worth
seeing.

## The monitor observes; the central brain decides

A colocated monitor used to be able to interrupt its worker. Measured, with
that lever: **six `ResultMessage`s, every one `aborted_*`, zero completed
turns** — the monitor interrupted its worker into failure and then judged it
for having been interrupted. Verdict 13 of that run, verbatim:

> *"The worker run was interrupted by the user before any work could be completed."*

The lever is gone. The monitor files verdicts through `store.judge`, which
needs no lease, so they reach the worker on its next wake and the central brain
through the `WorkerReport`. Nothing is lost; only the mid-flight shove is
withheld. Severity now means *confidence that something is wrong*, not *what to
do about it* — and the observation carries an `elapsed` field, because a judge
that cannot tell second three from minute ten grades every unfinished worker as
a failing one.

## Worker outcomes are a contract

A worker's process exit and its durable `WorkerReport` must agree. Exit alone
is never enough to infer completion:

| Exit | Meaning |
|---|---|
| `0` COMPLETED | validated report, `completed=True` |
| `10` INCOMPLETE | validated runtime, report published, completion declined |
| `65` INPUT_REJECTED | could not bind to its own durable assignment |
| `70` RUNTIME_FAILURE | runtime raised; no validated terminal report |
| `71` INFRA_FAILURE | stopped before the lease; no report |
| `72` SHUTDOWN_UNCONFIRMED | SDK child may still be live; branch handoff unsafe |
| `130` INTERRUPTED | cancelled |

The report carries `final_state_id`, `contract_digest`, `outputs`,
`monitor_severity`, `uncertain`, and `uncertainty_reasons` — enough for a
supervisor to decide without asking the worker anything.

## The substrate protects itself

A sub-brain's worktree *is* a branch head, so a worker running git takes the
branch out from under the layer that owns it. Both halves are enforced:

- **The jail refuses ref-movers.** `commit`, `reset`, `checkout`, `revert`, `rebase`, `stash`, `update-ref`, `gc` and friends are denied with a reason the model reads verbatim — *"the harness commits your work for you"*. Reading git (`status`, `diff`, `log`, `show`) and `git add` stay allowed. Verified live: a worker explicitly told to commit its work adapted and still completed in 35 s.
- **`ForeignHead` names what got through anyway.** A head carrying no state metadata, or a working tree that has detached from its branch, is refused on the write paths — never on reads, because whoever repairs it has to look at it first.

## Architecture

| Module | Thesis role | What it owns |
|---|---|---|
| [memstore/store.py](taste/memstore/store.py) | *Virtual memory* | States, branches, leases, rollback-as-append, typed merge, publication |
| [memstore/backend.py](taste/memstore/backend.py) | *Page tables* | Git plumbing only; nothing above it knows a git command |
| [brains/central_planner.py](taste/brains/central_planner.py) | *Scheduler* | Immutable plan revisions fenced to an observed world state |
| [brains/central_runtime.py](taste/brains/central_runtime.py) | *Reconciliation loop* | Crash-replayable cycles; plan → prepare → start → collect → deliver |
| [brains/supervisor.py](taste/brains/supervisor.py) | *Process table* | Fork handshake, descendant reaping, durable run records, delivery gate |
| [brains/worker_runtime.py](taste/brains/worker_runtime.py) | *Process* | One worker's SDK session, its checkpoints, its terminal report |
| [brains/monitor.py](taste/brains/monitor.py) | *Observability* | Batched judgement off the memstore tail; verdicts, never interventions |
| [brains/jail.py](taste/brains/jail.py) | *Memory protection* | Pre-tool veto: worktree containment and git ref protection |
| [brains/delivery.py](taste/brains/delivery.py) | *Transaction commit* | Project certified products into integration; conflicts are values |
| [kernel.py](taste/kernel.py) · [recovery.py](taste/recovery.py) | *Single-process line* | The original `taste run` kernel, its fault taxonomy and rollback |

## Quickstart

```bash
conda create -n agent-os python=3.11 -y && conda activate agent-os
pip install -e '.[dev,brains]'
export ANTHROPIC_API_KEY=sk-ant-...
```

**Hermetic, no API key** — the *step-87 scenario*: step 2 silently regresses,
the Monitor catches it with the project's own `pytest`, the kernel rolls back,
the retry lands clean, and the failed attempt leaves no trace on the branch.

```bash
python examples/refactor_demo/simulate.py
```

**Single-process line, real model:**

```bash
python examples/refactor_demo/bootstrap.py /tmp/refactor-demo
taste run "add type hints to legacy_math and split run() into small helpers" \
    --agent examples/refactor_demo/agent_desp.md --workspace /tmp/refactor-demo
```

Inspect with the same state the harness used — no sidecars, no out-of-band
files:

```bash
git -C /tmp/refactor-demo log taste/session-<id> --oneline --graph
git -C /tmp/refactor-demo show taste/session-<id>:.taste/plan.json
```

The central brain has **no CLI yet**; it is a library, driven as shown above.

## Tests

```bash
pytest -q          # 1,239 tests, none requiring an API key
```

| Layer | Tests | Load-bearing |
|---|---|---|
| memstore | 126 | atomicity under injected faults, lease exclusion, typed merge, foreign-head detection |
| brains | 448 | launch seam, worker terminal contract, monitor replay after SIGKILL, planner receipt durability |
| kernel | 69 | step-87 rollback, parallel waves, fault taxonomy, golden run signature |

CI runs `ruff` and the full suite on 3.11 and 3.12 with `.[dev,brains]`
installed. [`tests/golden.py`](tests/golden.py) reduces a run to a fingerprint —
event kinds, payload keys, commit chain, per-step outcomes, with SHAs and
timings excluded — so *"this subsystem changes nothing when disabled"* is one
assertion rather than a hope.

## Known gaps

Stated because a harness that hides its failure modes is the thing this repo
argues against. All of these are measured, not suspected.

- **`git reset --hard` inside a worktree is undetected.** It lands on a commit the layer *did* create, so the metadata check passes while the branch silently forks backwards. A forward-only check catches it and also refuses the coordinator's own documented recovery from a rewound control branch, so re-landing it needs a contract a repair path can opt out of.
- **Budget exhaustion escapes `cycle()` as an exception** rather than returning a `budget_blocked` status. An under-budgeted goal raises on the first cycle.
- **No give-up guard.** An impossible goal replans indefinitely; there is no maximum generation or abandonment rule.
- **`Store` is not thread-safe, and nothing says so in its API.** Its repo lock is `flock`-based and re-entrant within a process, so it excludes processes, not threads. Production runs one worker per process; a future in-process orchestrator would lose git notes.
- **Artifact-id collisions leave no record** of which claim integration accepted.
- **The catalog has writers but no readers.** `store.catalog()` and `store.search()` return real hits; nothing outside the planner's world snapshot consults them.
- **The jail is a gate, not a sandbox.** A command parser is defeated by `sh -c`, `eval`, or a variable. The real boundary is the SDK's OS-level sandbox underneath it.
- **Inter-agent communication is deliberately unbuilt.** Nothing should cross an agent boundary that a monitor has not passed.

## License

MIT — see [LICENSE](LICENSE).

## Credit

Thesis: [Beyond the Harness](https://zanwenfu.com/blog/agent_harness_blog).
Feedback, counter-examples and failure modes are wanted — open an issue or
reach [Zanwen Fu](mailto:zanwen.fu@duke.edu).
