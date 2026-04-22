# todo_api demo — the real Claude end-to-end run

A tiny Flask TODO API with a pytest suite. The agent's job is to add a non-trivial feature — validated priorities and sorted responses — without breaking the existing tests.

This example is the **credibility demo** for the repo: it's where a real Claude drives the Planner, Worker, and Monitor end-to-end.

## What it tests

The task — *"Add an integer `priority` field (1–5, default 3) to Item; validate it on POST; sort `GET /items` by priority descending; keep every existing test green and add new tests for validation + sort order"* — naturally decomposes into ~5 steps, each verifiable by `pytest -q`.

A successful run exercises:

- **Planner** producing a tight 4–6-step plan with `pytest` verifications.
- **Worker** editing `app.py` and `tests/test_app.py` across multiple steps, using `read_file` before `write_file`.
- **Monitor** running `pytest` after every step; failing checkpoints trigger rollback + retry.
- **Memory** growing one commit per step on a dedicated session branch.
- **Cache** hitting on repeated system prompts (visible in the final cost breakdown).

## Running it

```bash
# Make sure the repo's deps are installed (you've done this already):
pip install -e '.[dev]'
pip install flask              # the target app's runtime

# Put your key in .env at the repo root (gitignored):
#   ANTHROPIC_API_KEY=sk-ant-...

# Bootstrap a throwaway workspace:
python examples/todo_api/bootstrap.py /tmp/taste-todo

# Run:
taste run "Add an integer priority field (1-5, default 3) to Item; validate it on POST; sort GET /items by priority descending; keep every existing test green and add new tests for validation and sort order." \
    --agent examples/todo_api/agent_desp.md \
    --workspace /tmp/taste-todo
```

Expected cost envelope: **~$0.50–$2 per full run** depending on how many retries the Worker takes.

## What you'll see

- A live event stream showing Plan → Step × N → Monitor verdicts → (rollback?) → final status.
- A per-model usage table with calls, tokens, cache reads, and estimated USD cost.
- A clean git log on `taste/session-<id>` — one commit per step, verdicts bundled in.

Inspect afterwards:

```bash
git -C /tmp/taste-todo log taste/session-<id> --oneline --graph
git -C /tmp/taste-todo show taste/session-<id>:.taste/plan.json
cat /tmp/taste-todo/.taste/events.jsonl | jq .       # full event log
```

## Recorded run

See [`runs/`](runs/) for the transcript of a recorded run that lands in the repo. That's the artifact the main [README](../../README.md) links to as evidence that the harness works against a real model, not just against scripted workers.
