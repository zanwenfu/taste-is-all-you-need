# Refactor Demo

End-to-end flow for the Agent OS backbone.

## What the demo shows

1. The Planner decomposes "add type hints and split the monolithic function" into a small sequence of reversible steps.
2. A Worker executes each step on its own session branch, producing one commit per checkpoint.
3. The Monitor runs `pytest` after each step. If a step breaks the suite, the kernel does a `git reset --hard` to the pre-step checkpoint and asks the Worker to retry with the monitor's feedback injected as context.

## Quickstart (real Claude run)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/refactor_demo/bootstrap.py /tmp/refactor-demo
taste run "add type hints to legacy_math and split run() into small helpers while keeping all tests green" \
    --agent examples/refactor_demo/agent_desp.md \
    --workspace /tmp/refactor-demo
```

Inspect the session afterwards:

```bash
git -C /tmp/refactor-demo log taste/session-<id> --oneline --graph
git -C /tmp/refactor-demo show taste/session-<id>:.taste/plan.json
git -C /tmp/refactor-demo show taste/session-<id>:.taste/monitor/step-02.json
```

## Reproducible rollback story (no API key needed)

`pytest tests/test_kernel_rollback.py` runs a hermetic version of the demo using scripted workers. It reproduces the step-87 story in miniature:

- Step 2 is deliberately broken on the first attempt.
- The Monitor catches it; the kernel rolls back.
- A corrected retry passes.
- The final branch has no trace of the failed attempt except in the event log.
