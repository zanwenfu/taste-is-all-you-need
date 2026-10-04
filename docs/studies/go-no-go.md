# Go/no-go: does Taste's supervision help an unmodified agent?

Registered on 2026-10-04, before any go/no-go trial ran
([issue 26](https://github.com/zanwenfu/taste-is-all-you-need/issues/26)).
The code and settings are fixed at the commit tagged `go-no-go-1`.

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
4. Under Taste: agent runs stopped by their monitor, and submissions the
   certifier refused in trials the verifier passed.
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

(Filled in after the run: the commands, the report and the decision.)
