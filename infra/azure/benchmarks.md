# Running a Harbor benchmark with the Taste agent

`taste.benchmarks.harbor_agent:TasteAgent` runs one benchmark task as a goal in
Harbor's standard Docker environment. The task is run as published: its
container, user, working directory, network rule and time limit, with no
environment subclass, extra Compose file or override. The coordinator, its
workers and their monitors run on the host; only shell commands enter the task
container, one at a time.

This page is the procedure that has been run on the test VM. What has and has
not been validated is at the end.

## What the host needs

- Linux with cgroup v2 and systemd, Docker with the Compose and Buildx plugins,
  and nftables (Harbor's egress sidecar enforces a task's network rule).
- Harbor runs as **root**: the trial owner admits the task container through the
  local Docker socket and runs the goal in bounded systemd units.
- An unprivileged account with no supplementary groups (here `bugbash`) runs the
  goal's processes. It never gets the Docker socket.
- Two Python environments: Harbor's (`harbor==0.23.0`), and the worker's with
  this project's dependencies. Harbor's must be able to import `taste`
  (`PYTHONPATH`), from a checkout that root owns and the worker user can read.
- A root-only file (mode 600) with `AZURE_OPENAI_BASE_URL` and
  `AZURE_OPENAI_API_KEY`. The key reaches the goal as a private systemd
  credential. It never enters the task container, a prompt or a trajectory.

## Run

    sudo infra/azure/run-harbor.sh <job> <tasks dir> [harbor run options]

    # errata-bench: Harbor must also find its agents and task digests
    sudo EXTRA_PATH=/root/errata/errata-bench/src infra/azure/run-harbor.sh \
        errata-astra /root/errata/v1.0.2-dataset/harbor -k 3 -n 2 --max-retries 2

The script starts `harbor run -a taste.benchmarks.harbor_agent:TasteAgent
-m azure/gpt-6-astra` in one systemd unit (`taste-harbor-<job>`). Follow it
with `journalctl -u taste-harbor-<job> -f`. Results are under the jobs
directory, as for any Harbor agent.

The agent's own processes run on the host and call only the Azure OpenAI
endpoint; it has no web or search tool. Every command it runs in the task
container is subject to the task's network rule, which Harbor enforces there.

## Settings (`--ak name=value`)

These are the run's disclosed configuration; each trial records them in its
result metadata (`agent_result.metadata.taste.configuration`).

| Setting | Default | Meaning |
| --- | --- | --- |
| `worker_model` | the trial's model | Served model for workers and monitors (`gpt-6-astra` or `gpt-6-sol`). The coordinator is always `gpt-6-astra`. |
| `worker_effort` | `low` | Worker reasoning effort: `low`, `medium`, `high`. Monitors use `low`. |
| `planner_effort` | `medium` | The coordinator's reasoning effort. Empty leaves the provider's default, which is close to none. |
| `spend_cap_usd` | 15 | What one trial may really spend before it stops and replies. |
| `worker_spend_cap_usd`, `monitor_spend_cap_usd` | 6, 2 | The same, for one worker and its monitor. |
| `max_assignments` | 1 | Assignments one plan may hold. One container and a serial terminal make one worker at a time the honest default. |
| `reply_reserve_seconds` | 210 | Held back from working time for the coordinator's closing reply. |
| `handoff_seconds` | 150 | Held back after that for settlement, the record and the handoff. |
| `command_seconds` | 600 | Longest timeout one command may ask for (default per command: 120). |
| `agent_timeout_sec` | from `task.toml` | Only if the task's published agent time must be overridden for a drill. |

Admission budgets are larger than the spend caps: a call is admitted only if
the role's spending plus that call's worst case fits its budget, so each
budget is the cap plus one worst-case call. The caps are what limit real
spending.

## What a trial leaves

- `<job>/<trial>/agent/trajectory.json`: the instruction, every worker step in
  the order it happened (one tool call per step, marked as a sidechain), and the
  coordinator's final reply. `extra.audit_flags` lists what this system's own
  audit lacks for that trial. An empty list means none.
- `/var/lib/taste-trials/<token>/`: the private record (root only): the complete
  settled trajectory with planner and monitor calls, the terminal ledger, every
  model call's journal and the goal's memory. `metadata.taste.trial` in the
  Harbor result names the token.

A trial is handed to the benchmark's verifier whatever the audit found. A run
that stops on time, budget, generation bound or planner failures still ends
with the coordinator's closing reply; one cancelled by the benchmark's own
time limit is sealed first and graded as a timeout.

## errata-bench

    # 1. unpaid: which trials can be graded, and why not the others
    cd /root/errata/errata-bench && /root/errata/grade-venv/bin/python scripts/grade_harbor.py \
        /root/errata/v1.0.2-dataset /root/errata/jobs/<job> --out /root/errata/runs/<job> \
        --admission /root/errata/v1.0.2-dataset/admission/gpt-6-astra --rows-only

Grading itself is paid and runs in errata-bench's own environment with the
judge's key (see errata-bench's running guide). Taste's reply is written by
gpt-6-astra, which is also errata-bench's judge, so its self-grading check
refuses unless `ERRATA_ALLOW_SELF_GRADING=1` is set; results graded that way
must be reported as self-graded.

The dataset on the VM was copied from the local v1.0.2 release folder and
every file except `README.md` verified against the published `SHA256SUMS`.

## Validated so far

- errata-bench's own `StandIn` agent on this VM: image build, egress rule,
  shared verifier, `answer.json`.
- The complete goal with real gpt-6 models on synthetic tasks: completion, a
  run out of time (closing reply), a hanging test.

Not yet validated: more than one trial at a time, a full three-attempt run,
Terminal-Bench's separate verifier, and any environment other than local Docker.
