# Recorded run — `flask_feature_agent` on the TODO API

A real [Claude Sonnet 4.6](https://www.anthropic.com/claude) run, not a scripted replay. The Planner, Worker, and Monitor are all the actual models. The artifact below is *exactly* what the kernel printed plus the git state it produced.

## Environment

| | |
|---|---|
| Agent | [`flask_feature_agent`](../agent_desp.md) |
| Models | Planner: `claude-opus-4-7` · Worker / Monitor: `claude-sonnet-4-6` |
| Workspace | `/tmp/taste-todo-realrun-v2` (bootstrapped from [`template/`](../template)) |
| Session | `taste/session-polished` |
| Task | *"Add an integer priority field (1–5, default 3) to Item; validate it on POST; sort GET /items by priority descending; keep every existing test green and add new tests for priority validation and sort order."* |
| Max retries | 1 |

## Result

| Metric | Value |
|---|---|
| Status | **completed** |
| Elapsed | 43.13 s |
| Estimated cost | **$0.0964** |
| Model calls | 7 (all Sonnet 4.6) |
| Input / output tokens | 16 545 / 3 117 |
| Cache hit rate | 0.0% *(see "observations")* |
| Final pytest | **15 / 15 passing** |
| Rollbacks | 0 — the plan landed on the first attempt |

## Plan (produced by the Planner, committed as `.taste/plan.json`)

```json
{
  "task": "Add an integer priority field (1-5, default 3) to Item; validate it on POST; sort GET /items by priority descending; ...",
  "steps": [
    {
      "id": "step-01",
      "description": "Update app.py to: (1) add `priority: int = 3` field to Item dataclass, (2) validate priority on POST (must be int 1-5, default 3 if omitted, reject if out of range), (3) sort GET /items by priority descending.",
      "verification": { "kind": "shell", "command": "pytest -q tests/test_app.py" }
    },
    {
      "id": "step-02",
      "description": "Add new tests to tests/test_app.py covering: priority field present in response with default 3, custom priority stored correctly, priority out-of-range (0 and 6) rejected with 400, non-integer priority rejected with 400, GET /items returns items sorted by priority descending.",
      "verification": { "kind": "shell", "command": "pytest -q tests/test_app.py" }
    }
  ]
}
```

## Live event stream (from the CLI)

```
>> task=... session=polished branch=taste/session-polished agent=flask_feature_agent
PLAN steps=2
STEP id=step-01 attempt=1
WORK tools=2 stop=end_turn
EVAL id=step-01 passed=True reason=`pytest -q tests/test_app.py` exited 0 sha=c30cfc4
STEP id=step-02 attempt=1
WORK tools=3 stop=end_turn
EVAL id=step-02 passed=True reason=`pytest -q tests/test_app.py` exited 0 sha=b00412a
DONE status=completed elapsed=43.13 cost_usd=0.0964 cache_hit_rate=0.0
```

## Final git log on the session branch

```
b00412a step-02: Add new tests to tests/test_app.py covering: priority field present in response with default 3, custom priority stored correctly, priority out-of-range (0 and 6) rejected with 400, non-integer priority rejected with 400, GET /items returns items sorted by priority descending. [Monitor: PASS]
c30cfc4 step-01: Update app.py to: (1) add `priority: int = 3` field to Item dataclass, (2) validate priority on POST (must be int 1-5, default 3 if omitted, reject if out of range), (3) sort GET /items by priority descending. [Monitor: PASS]
9617a9c plan: commit decomposition
7c5b3d0 initial: flask todo api + tests
```

Every step is one commit. The Monitor's verdict is bundled into the commit message — the git log *is* the audit trail, no sidecar.

## What the Worker actually changed

<details>
<summary><code>app.py</code> (Item dataclass + POST validation + sorted GET)</summary>

```diff
@@ class Item:
     id: int
     title: str
     done: bool = False
+    priority: int = 3


@@ def list_items():
-        return [asdict(x) for x in _STORE.values()]
+        sorted_items = sorted(_STORE.values(), key=lambda x: x.priority, reverse=True)
+        return [asdict(x) for x in sorted_items]


@@ def create_item():
-        item = Item(id=_NEXT_ID, title=title.strip())
+        # Validate priority: default 3 if omitted, must be an int in [1, 5]
+        if "priority" not in body:
+            priority = 3
+        else:
+            priority = body["priority"]
+            if not isinstance(priority, int) or isinstance(priority, bool):
+                return {"error": "priority must be an integer between 1 and 5"}, 400
+            if priority < 1 or priority > 5:
+                return {"error": "priority must be an integer between 1 and 5"}, 400
+
+        item = Item(id=_NEXT_ID, title=title.strip(), priority=priority)
```
</details>

Notably, the Worker caught the `isinstance(priority, int)` gotcha — `bool` is a subclass of `int` in Python, so a naive type check would pass `True` as a valid priority. It added an explicit `isinstance(priority, bool)` check to reject it. That's a real piece of Python subtlety the harness preserved through the Monitor.

## Test output on the final tree

```
tests/test_app.py::test_list_is_empty_initially PASSED                   [  6%]
tests/test_app.py::test_create_and_list PASSED                           [ 13%]
tests/test_app.py::test_create_rejects_missing_title PASSED              [ 20%]
tests/test_app.py::test_create_rejects_blank_title PASSED                [ 26%]
tests/test_app.py::test_delete_removes_item PASSED                       [ 33%]
tests/test_app.py::test_delete_missing_item_404 PASSED                   [ 40%]
tests/test_app.py::test_priority_defaults_to_3 PASSED                    [ 46%]
tests/test_app.py::test_custom_priority_stored_correctly PASSED          [ 53%]
tests/test_app.py::test_priority_zero_rejected PASSED                    [ 60%]
tests/test_app.py::test_priority_six_rejected PASSED                     [ 66%]
tests/test_app.py::test_non_integer_priority_rejected PASSED             [ 73%]
tests/test_app.py::test_float_priority_rejected PASSED                   [ 80%]
tests/test_app.py::test_boolean_priority_rejected PASSED                 [ 86%]
tests/test_app.py::test_get_items_sorted_by_priority_descending PASSED   [ 93%]
tests/test_app.py::test_get_items_same_priority_all_returned PASSED      [100%]

============================== 15 passed in 0.07s ==============================
```

9 new tests added by the Worker on top of 6 starter tests — all green.

## Observations (honest readout, not marketing)

What this run proves:
- The harness runs end-to-end against a real Claude: Planner → Worker × N → Monitor → commit, with the real-model variant of every core.
- The Planner's verification taste is **tunable**. An earlier version of this same task — before we tightened `PLANNER_SYSTEM` to ban `grep` checks — produced a 3-step plan where two intermediate steps verified with `grep priority app.py` instead of `pytest`. The code was still correct because Sonnet is strong, but the Monitor was effectively asleep for those steps. After the prompt change, every verification runs the actual test suite. The failure mode and the fix both live in the repo's git history.
- Cost and latency on a real task sit at **~$0.10 and ~45 s** for a clean 2-step landing. Retries and longer plans scale it up predictably.

What this run does **not** yet prove:
- **Caching benefit.** Cache hit rate is 0%. Anthropic prompt caching has a ~1024-token minimum per breakpoint for Sonnet; our current system blocks are shorter than that. This is a known prompt-engineering follow-up — consolidate the static system prefix until it crosses the threshold — not an architectural issue.
- **Long-horizon rollback.** The plan was 2 steps and landed clean. The step-87-style story is demonstrated hermetically in `tests/test_kernel_rollback.py`; a natural real-model rollback would need either a harder task or a deliberate adversarial workspace. On the roadmap.
- **Parallel execution.** This is single-worker. The multi-core thesis gets exercised in milestone B (worktree-backed parallel workers).

## Reproducing this run

```bash
pip install -e '.[dev]' && pip install flask
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
python examples/todo_api/bootstrap.py /tmp/taste-todo
taste run "Add an integer priority field (1-5, default 3) to Item; validate it on POST; sort GET /items by priority descending; keep every existing test green and add new tests for priority validation and sort order." \
    --agent examples/todo_api/agent_desp.md \
    --workspace /tmp/taste-todo \
    --max-retries 1
```

Exact numbers will drift run-to-run because the Planner and Worker are sampling; the shape (2–3 steps, all pytest-verified, $0.05–$0.20) is stable.
