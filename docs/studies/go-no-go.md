# Go/no-go: does Taste's supervision help an unmodified agent?

Registered on 2026-10-04, before any go/no-go trial ran
([issue 26](https://github.com/zanwenfu/taste-is-all-you-need/issues/26)).
The code and settings are fixed at the commit tagged `go-no-go-3` (see the
amendments below; `go-no-go-1` was the first registration, and the first run
used `go-no-go-2`).

## Question

On tasks the study may tune on, does mini-swe-agent run under Taste's
planner, monitors and certifier solve at least as many tasks as the same agent
run alone, at the same cost cap? The answer decides whether this supervision
is sound enough to build rollback of the task environment on. It is not a
result to report: the tasks are tuning tasks, chosen so that what is learned
here can be acted on.

## Agent and model

- mini-swe-agent 2.4.6 with its own `mini` configuration (the one Harbor runs
  it with on Terminal-Bench), its loop, prompts, bash tool and parsing
  unchanged (`taste/agents/mini_swe_agent.py`).
- GPT-5.6 Luna (version 2026-07-09) on Azure in every role: the agent and,
  under Taste, the coordinator and the monitors. Reasoning effort: the agent
  low, the coordinator medium, the monitors low.

## Arms

| Arm | Settings | What runs |
| --- | --- | --- |
| alone | `--ak agent=mini-swe-agent --ak services=none --ak reply_reserve_seconds=10` | One run of the task as given. A fixed rule plans (the task verbatim, one assignment); no monitor; nothing certifies. (Reserve: see the amendment below.) |
| taste | `--ak agent=mini-swe-agent` | Taste's coordinator plans, a monitor judges each step, the certifier checks each report. A monitor judgement of wrong or lost stops the agent; the coordinator may run it again with a new assignment. |

Both arms run through the same worker process, model journal, terminal broker
and settled record. Only the services differ.

## Cost and limits

Both arms run with the same settings, beyond those that define the arm:

    --ak spend_cap_usd=2 --ak worker_spend_cap_usd=2 --ak monitor_spend_cap_usd=1
    --ak worker_max_calls=1000 --ak monitor_max_calls=400 --ak max_commands=2000

A trial takes on no new work once it has spent $2. Alone, the agent may spend
all of it; under Taste, the coordinator, the agent runs and their monitors
share it, though a run already started may finish past it. mini-swe-agent's
own cost limit ($3) applies inside both. On this model no trial is expected to
come near these caps (the drills spent $0.004 alone and $0.03 under Taste), so
the arms are compared on what each actually spent (measure 3); the caps only
bound a run that goes wrong.

The call and command limits are set high so that the task's own time limit,
not Taste's defaults (tuned on shorter tasks), ends a long run, as it would
for the agent run by itself. One limit is not lifted: a model request may hold
at most 1 MiB, so a very long conversation ends there in either arm. The
report counts runs that ended for that reason.

## Tasks and runs

- The 20 tuning tasks of Terminal-Bench 2.1
  (`data/splits/terminal-bench-2-1.json`), each with its own published time
  limit and resources, in Harbor's local Docker environment on one Azure VM,
  two trials at a time.
- Three trials per task per arm: 120 trials.

## Measures

Produced by `scripts/go_no_go_report.py` from the two job directories:

1. Task-runs solved per arm (the task's verifier gives reward 1).
2. Per task, the share of its runs each arm solved; the tasks where each arm
   did better; an exact two-sided sign test over the tasks that differ.
3. Dollars per trial and per solved task, by role (coordinator, agent,
   monitor).
4. Under Taste: agent runs stopped by their monitor, submissions the
   certifier refused in trials the verifier passed, and goals Taste closed as
   complete that the verifier failed.
5. Trials with audit flags or exceptions, and their causes.

## Decision

- **Go** if the taste arm solves at least as many task-runs as the alone arm,
  less one. Rollback is then built on this supervision.
- **Fix first** otherwise: find the causes on these tuning tasks, fix them,
  and run this go/no-go again before building rollback.

Either way, defects the run shows (wrong refusals, needless stops, failed
records) are fixed before the study's settings are frozen.

## Budget and stopping

At most $100 of model spend. If the first 20 trials project past that, the
run stops and the decision returns to the project owner. A defect that stops
trials from running at all stops the run; it restarts from the beginning once
fixed, with the fix recorded below.

## Record

**Amendment before the run (2026-10-04).** Both arms held back 210 seconds of
each task's time for the coordinator's closing reply. The supervised arm needs
them: its closing reply is a model call. The fixed plan's closing takes no
time, so in the alone arm those seconds were only taken from the agent (on a
15-minute task, from about 12.5 minutes of work to 9). That reserve is part of
what supervision costs, so the alone arm runs with the least the goal admits:
`--ak reply_reserve_seconds=10`. Found while reviewing a six-trial pilot that
used the registered settings; no go/no-go trial had run.

**Second amendment before the run (2026-10-04): code at `go-no-go-2`.** Two
pilots of six trials (three tuning tasks, both arms, the registered settings)
found the supervised arm's planner refused for bookkeeping, not for its
decisions. On GPT-5.6 Luna: a request id mis-copied, a schema name left out,
an assignment's caps or routing changed, a metadata field missing, an input
handed to an agent that reads none, and replies that were not valid JSON. Four
refusals in a row end a goal, and two of the six supervised trials ended so. In
`go-no-go-2` the harness writes what the request and the policy fix and records
what it wrote with the plan, plans are asked for as one JSON object (which the
service enforces), and a hosted agent's assignment is given no inputs. The
planner's decisions are validated as before. Also measured now: goals Taste
closed as complete that the verifier failed. No go/no-go trial had run.

**First run, at `go-no-go-2` (2026-10-04, 10:39 to 21:32 UTC).** One Azure VM
with 4 vCPUs, two trials at a time (one once the alone arm had finished, at
12:40). From the tagged code:

    MODEL=azure/gpt-5.6-luna infra/azure/run-terminal-bench.sh gng1-alone tuning -k 3 -n 1 \
        --ak agent=mini-swe-agent --ak spend_cap_usd=2 --ak worker_spend_cap_usd=2 \
        --ak monitor_spend_cap_usd=1 --ak worker_max_calls=1000 --ak monitor_max_calls=400 \
        --ak max_commands=2000 --ak services=none --ak reply_reserve_seconds=10
    MODEL=azure/gpt-5.6-luna infra/azure/run-terminal-bench.sh gng1-taste tuning -k 3 -n 1 \
        --ak agent=mini-swe-agent --ak spend_cap_usd=2 --ak worker_spend_cap_usd=2 \
        --ak monitor_spend_cap_usd=1 --ak worker_max_calls=1000 --ak monitor_max_calls=400 \
        --ak max_commands=2000
    python scripts/go_no_go_report.py --arm alone=/root/tb/jobs/gng1-alone \
        --arm taste=/root/tb/jobs/gng1-taste

| Arm | Solved | Trials | $/trial | $/solved | Coordinator $ | Agent $ | Monitor $ | Monitor stops | Refused, verifier passed | Flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alone | 17 | 60 | 0.005906 | 0.020844 | 0.0 | 0.354354 | 0.0 | 0 | 0 | 0 |
| taste | 24 | 60 | 0.239512 | 0.598781 | 6.798815 | 2.647677 | 4.924252 | 46 | 81 | 8 |

Paired by task (20 tasks): taste better on 7, alone better on 1
(kv-store-grpc), same on 12; exact sign test p = 0.07.

Measure 4, supervised arm: 46 agent runs stopped by their monitor;
81 submissions refused in trials the verifier passed (an upper bound on
wrong refusals: a later run may have changed the container); 6 of
the 20 goals closed as complete failed the verifier. Measure 5: 8
trials flagged, four `stopped:planner_failed` and the rest
`evidence_incomplete` from work cut off at a trial's end
([#48](https://github.com/zanwenfu/taste-is-all-you-need/issues/48)); no
exceptions. Model spend: $14.72 ($0.35 alone, $14.37 supervised) in all.

**Decision on the first run: Go** by the rule. The supervised arm solved 24 task-runs and the
alone arm 17 (the rule asks at least 16). It did better on 7 of the 20
tasks and worse on one (kv-store-grpc, 2 of 3 against 3 of 3). It cost about
40 times as much per trial ($0.24 against $0.006) and 29 times as much per
solved task ($0.60 against $0.021): the coordinator $6.80 of its $14.37, the
monitors $4.92, the agent's own calls $2.65. Whether the gain comes from
supervision or from running the agent more than once is what the study's
budget-matched controls
([#34](https://github.com/zanwenfu/taste-is-all-you-need/issues/34)) are for.

**Third amendment, after the first run (2026-10-04): code at `go-no-go-3`.**
The first run's records showed defects in the supervised arm, each tracked
as an issue and fixed with tests while the run went on (the run itself used
`go-no-go-2` throughout):
[#40](https://github.com/zanwenfu/taste-is-all-you-need/issues/40) the
coordinator, monitors and certifier looked for a container task's files in
memory, where they never are, so certified work was assessed "not met" and
finished goals kept running;
[#41](https://github.com/zanwenfu/taste-is-all-you-need/issues/41) contracts
asked the hosted agent to write its own report, and its monitor judged it;
[#42](https://github.com/zanwenfu/taste-is-all-you-need/issues/42) refused
plans were counted over the whole goal, and several bookkeeping slips were
still refusals;
[#43](https://github.com/zanwenfu/taste-is-all-you-need/issues/43) the step
monitor judged a run's last batch without its earlier work;
[#44](https://github.com/zanwenfu/taste-is-all-you-need/issues/44) a goal was
closed on a documented test failure;
[#45](https://github.com/zanwenfu/taste-is-all-you-need/issues/45) agents were
stopped for defects they were still fixing;
[#46](https://github.com/zanwenfu/taste-is-all-you-need/issues/46)
certification failed closed when a finding's hash was mis-copied.
Each change to a prompt was first checked by replaying recorded decisions
with the same model and settings (`scripts/replay_decision.py`; the code that
decided rebuilds the prompt it sent, byte for byte), and the changes together
by two pilots on tuning tasks (nine trials).
The go/no-go is run again at `go-no-go-3`: both arms, the same tasks, runs
and settings. So that it takes hours rather than a day, the VM is resized to
32 vCPUs (Standard_D32as_v7) and each arm runs eight trials at a time; the
first run had two at a time in all. The decision below is taken on this
second run; the first run's result stands as recorded.

(Filled in after the second run: its commands, report and the decision.)
