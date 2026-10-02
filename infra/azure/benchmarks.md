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
| `planner_effort` | `medium` | The coordinator's reasoning effort. Empty leaves the provider's default. Measured on gpt-6-astra, the level changes little: the model reasons briefly at every level. |
| `spend_cap_usd` | 15 | What one trial may really spend before it stops and replies. |
| `worker_spend_cap_usd`, `monitor_spend_cap_usd` | 6, 2 | The same, for one worker and its monitor. |
| `max_assignments` | 1 | Assignments one plan may hold. One container and a serial terminal make one worker at a time the honest default. |
| `reply_reserve_seconds` | 210 | Held back from working time for the coordinator's closing reply. |
| `plan_seconds` | 90 | No new plan is started with less working time than this; the run closes instead. |
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

## Validated so far (2026-10-01, test VM, gpt-6-astra in every role)

Fifteen trials of four errata-bench v1.0.2 tasks, one to five at a time.
errata-bench's unpaid admission check (`--rows-only`) counts all fifteen as
gradable; the fourteen run with the official time limit are counted official.

| What was exercised | Result |
| --- | --- |
| Completion, small instruction (2 to 10 KB), three tasks | 80 to 144 s, $0.89 to $1.72, 2 plans, no audit flags |
| Completion, large instruction (117 KB), twice | 263 and 275 s, $4.27 and $4.46, 2 plans, 13 and 14 commands, no audit flags |
| Two attempts of one task at once; five trials at once | no interference |
| The benchmark's own time limit cancels the agent (drill) | sealed within 5 s, `AgentTimeoutError`, commands recorded in order, graded as no answer |
| Working time runs out (drill) | worker stops in good order, exact cost, closing reply names what was run and what was not verified |
| A command that ignores TERM and leaves a detached child (`scripts/check_docker_terminal.py`, no model) | ended alone in 1.1 s, container and terminal still usable |

The first run of each kind found a defect, and each was fixed and run again;
the commit messages from 2026-10-01 describe them one by one. The largest was
a certifier that could not read a long run's evidence: the same 117 KB task
cost $18.58 and took 17 minutes before that fix.

Most errata-bench instructions are large: the median is 75 KB and 32 of the
55 tasks exceed 64 KB. A full run of 165 trials at the costs above is in the
region of $500 to $900 for the agent, and its spend caps bound it near
$3,000. Paid grading is separate.

Not validated: a full three-attempt run, paid grading, more than five trials
at once on a larger machine, Terminal-Bench's separate verifier, and any
environment other than local Docker.

Older fixture scripts under `scripts/check_*.py`, other than
`check_docker_terminal.py`, predate command-scoped timeouts and always-grade
settlement and have not been re-run.
