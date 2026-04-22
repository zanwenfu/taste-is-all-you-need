# parallel_demo — three workers, three worktrees, one merge

Three independent Python utility modules + pytest suite. The agent's job is to annotate all three with type hints. Because the modules are disjoint, the Planner can (and does) schedule the annotations as a parallel wave — three workers run concurrently on separate git worktrees, and the Orchestrator merges their branches back into the session when they all finish.

This is the *multi-core CPU* demo for the Agent OS.

## The architecture this exercises

1. **`Step.depends_on`** drives the DAG. The Planner emits a plan where steps 2/3/4 share `depends_on: ["step-01"]`, producing a single parallel wave of size 3.
2. **`Memory.add_worktree`** spawns a physical sibling worktree per parallel step. Each worker has its own working tree, its own branch, and its own file writes.
3. **`Kernel._run_wave`** fires up a `ThreadPoolExecutor` with one worker per step. They run concurrently — real parallelism, not async concurrency — because each worker is I/O-bound on subprocess `pytest` and on API calls.
4. **`Memory.merge_branch`** collapses each worker branch back into the session. If any worker failed, no merges happen (atomic wave commit). If any merge conflicts, `MergeConflict` halts the run with a structured signal.

## Recorded run

See [`runs/parallel.md`](runs/parallel.md) for the full transcript. Highlights:

- 4 steps, 2 waves (1 parallel wave of 3), completed in **21.46 s** wall-clock.
- **$0.0932** total, 12 Sonnet 4.6 calls.
- Git topology shows three diamond merges landing on top of step-01:

```
*   merge: step-04 from taste-wt/...-step-04
|\
| * step-04: Add type hints to list_utils.py [Monitor: PASS]
* |   merge: step-03 from taste-wt/...-step-03
|\ \
| * | step-03: Add type hints to string_utils.py [Monitor: PASS]
| |/
* |   merge: step-02 from taste-wt/...-step-02
|\ \
| |/
|/|
| * step-02: Add type hints to math_utils.py [Monitor: PASS]
|/
* step-01: No-op shared prep step [Monitor: PASS]
```

Dashboard: [`runs/dashboard.html`](runs/dashboard.html) ([screenshot](../../docs/img/dashboard-parallel.png)).

## Reproducing the run

```bash
export ANTHROPIC_API_KEY=sk-ant-...         # or put it in .env
python examples/parallel_demo/bootstrap.py /tmp/taste-parallel
taste run "Add Python type hints to math_utils.py, string_utils.py, and list_utils.py. REQUIRED: steps 02/03/04 each annotate one module and all declare depends_on=['step-01'] so they run IN PARALLEL on separate worktrees." \
    --agent examples/parallel_demo/agent_desp.md \
    --workspace /tmp/taste-parallel \
    --max-retries 1
taste dashboard --workspace /tmp/taste-parallel
```

## Hermetic smoke test

`pytest tests/test_parallel.py` runs the same parallel scheduler against scripted workers (no API needed) and asserts:

- `Plan.waves()` computes the right DAG topology.
- Three parallel workers land on three distinct filesystem paths.
- Atomic wave semantics — if any worker fails, no merges happen.
- The JSONL event log captures every `wave.begin`/`worktree.open`/`worktree.merge`/`wave.done`.
