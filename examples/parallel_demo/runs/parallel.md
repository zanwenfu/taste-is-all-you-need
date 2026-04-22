# Recorded run — `parallel_type_hint_agent` across three worktrees

A real Claude run where the Planner produces a DAG, the Kernel spawns a worktree per parallel step, three Workers execute **concurrently**, and the Orchestrator merges all three branches back into the session.

This is the multi-core story the blog argues for — made executable.

## Environment

| | |
|---|---|
| Agent | [`parallel_type_hint_agent`](../agent_desp.md) |
| Models | Planner: `claude-opus-4-7` · Worker / Monitor: `claude-sonnet-4-6` |
| Workspace | `/tmp/taste-parallel-realrun-v2` (bootstrapped from [`template/`](../template)) |
| Session | `taste/session-parfast` |
| Task | *Annotate three independent utility modules with type hints in parallel worker waves.* |
| Max retries | 1 |

## Result

| Metric | Value |
|---|---|
| Status | **completed** |
| Elapsed | **21.46 s** *(vs ~32 s sequential — the parallel wave ran 3 workers in ~8.5 s)* |
| Estimated cost | **$0.0932** |
| Model calls | 12 (all Sonnet 4.6) |
| Steps / Waves | 4 steps, 2 waves — **1 parallel wave of size 3** |
| Final pytest | **9 / 9 passing** |

## Plan the Planner produced

```json
{
  "steps": [
    {"id": "step-01", "description": "No-op shared prep step; verify baseline tests are green",
     "verification": {"kind": "shell", "command": "pytest -q"},
     "depends_on": []},
    {"id": "step-02", "description": "Add type hints to math_utils.py",
     "verification": {"kind": "shell", "command": "pytest -q tests/test_math_utils.py"},
     "depends_on": ["step-01"]},
    {"id": "step-03", "description": "Add type hints to string_utils.py",
     "verification": {"kind": "shell", "command": "pytest -q tests/test_string_utils.py"},
     "depends_on": ["step-01"]},
    {"id": "step-04", "description": "Add type hints to list_utils.py",
     "verification": {"kind": "shell", "command": "pytest -q tests/test_list_utils.py"},
     "depends_on": ["step-01"]}
  ]
}
```

`depends_on` is what turns a flat plan into a DAG. Three steps sharing `depends_on: ["step-01"]` become one parallel wave.

## Live event stream (excerpt)

```
>> task=... session=parfast branch=taste/session-parfast agent=parallel_type_hint_agent
PLAN steps=4 waves=2 parallel_waves=1
STEP id=step-01 attempt=1         # wave 1 — sequential bootstrap
EVAL id=step-01 passed=True       reason=`pytest -q` exited 0  sha=d048629

WAVE.BEGIN steps=['step-02', 'step-03', 'step-04'] size=3    # wave 2 — parallel
WORKTREE.OPEN step=step-02 branch=taste-wt/…/step-02 path=.taste-worktrees/…
WORKTREE.OPEN step=step-03 branch=taste-wt/…/step-03 path=.taste-worktrees/…
WORKTREE.OPEN step=step-04 branch=taste-wt/…/step-04 path=.taste-worktrees/…
STEP id=step-02 attempt=1   STEP id=step-03 attempt=1   STEP id=step-04 attempt=1
WORK id=step-02 tools=2     WORK id=step-03 tools=2     WORK id=step-04 tools=2
EVAL id=step-02 passed=True reason=`pytest -q tests/test_math_utils.py` exited 0
EVAL id=step-03 passed=True reason=`pytest -q tests/test_string_utils.py` exited 0
EVAL id=step-04 passed=True reason=`pytest -q tests/test_list_utils.py` exited 0
WORKTREE.MERGE step=step-02 source=taste-wt/…/step-02 sha=1115252
WORKTREE.MERGE step=step-03 source=taste-wt/…/step-03 sha=6d6cd5f
WORKTREE.MERGE step=step-04 source=taste-wt/…/step-04 sha=39a5c92
WAVE.DONE  steps=['step-02', 'step-03', 'step-04'] size=3
DONE status=completed elapsed=21.46 cost_usd=0.0932
```

## Git topology (the picture the thesis has been chasing)

```
$ git -C /tmp/taste-parallel-realrun-v2 log --graph --oneline --all taste/session-parfast
*   39a5c92 merge: step-04 from taste-wt/taste-session-parfast-step-04
|\
| * d3187da step-04: Add type hints to list_utils.py ... [Monitor: PASS]
* |   6d6cd5f merge: step-03 from taste-wt/taste-session-parfast-step-03
|\ \
| * | f35af81 step-03: Add type hints to string_utils.py ... [Monitor: PASS]
| |/
* |   1115252 merge: step-02 from taste-wt/taste-session-parfast-step-02
|\ \
| |/
|/|
| * e3f2118 step-02: Add type hints to math_utils.py ... [Monitor: PASS]
|/
* d048629 step-01: No-op shared prep step ... [Monitor: PASS]
* a4e0fb9 plan: commit decomposition
* 1bc8056 initial: three independent utility modules + tests
```

Three diamond merges — one per parallel worker branch — landing on top of the shared step-01 bootstrap. This is "multi-core CPU" for agents: different address spaces (worktrees), independent verifications, structured merge at the end.

## What the Workers actually produced

One worker per module, each editing only inside its own worktree:

- `math_utils.py`: `add(a: float, b: float) -> float`, `mul(a: float, b: float) -> float`, `clamp(x: float, lo: float, hi: float) -> float`
- `string_utils.py`: `upper(s: str) -> str`, `snake_case(s: str) -> str`, `truncate(s: str, n: int) -> str`
- `list_utils.py`: generic `head[T]`, `tail[T]`, `chunked[T]` via `TypeVar`

Every original test still passes on the merged session branch.

## Observations

What this run proves:
- The Planner can produce a DAG. Given a task flagged as parallelizable, it correctly emitted three steps sharing `depends_on: ["step-01"]`, creating one parallel wave of size 3.
- The Kernel's wave scheduler runs them concurrently on **physically separate git worktrees** — no shared working tree, no filesystem collision.
- Wall-clock drops from ~32 s (sequential) to **21.46 s** for the same work. The three worker-ops happened in ~8.5 s of real time.
- After the wave, three branches merged back into the session cleanly. The git graph shows the multi-core topology.
- The event stream captures every primitive (`wave.begin`, `worktree.open`, `worktree.merge`, `wave.done`) — the dashboard renders them as-is.

What this run doesn't yet prove:
- **Merge-conflict recovery at scale.** The demo chose disjoint modules by design. A real conflicting parallel run raises `MergeConflict`; the Kernel halts without committing — that's tested hermetically in `tests/test_memory.py::test_merge_conflict_raises_typed_exception` but isn't exercised on a real model.
- **Autonomous parallelism selection.** Claude picked parallel here because the task *told it to*. Without that hint the earlier run chose sequential (a deliberately conservative default). Making the Planner spot parallel opportunities on its own is prompt-engineering territory.

## Reproducing

```bash
pip install -e '.[dev]'
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
python examples/parallel_demo/bootstrap.py /tmp/taste-parallel
taste run "Add Python type hints to math_utils.py, string_utils.py, and list_utils.py ... steps 02/03/04 MUST all declare depends_on=['step-01'] so they run IN PARALLEL on separate git worktrees ..." \
    --agent examples/parallel_demo/agent_desp.md \
    --workspace /tmp/taste-parallel \
    --max-retries 1
taste dashboard --workspace /tmp/taste-parallel
```
