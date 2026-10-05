# Continue control: does trying again explain the gain?

Registered on 2026-10-05, before any trial of this arm ran
([issue 34](https://github.com/zanwenfu/taste-is-all-you-need/issues/34)).
The code and settings are fixed at the commit tagged `continue-control-1`.

## Question

In the go/no-go's second run (`docs/studies/go-no-go.md`), the agent under
Taste solved 25 of 60 task-runs and the agent alone 20 of 60. Taste runs the
agent again when its work is not done. Does running the agent again, with no
planning, monitoring or checking, do as well?

## Arm

`continue`: the go/no-go's settings for the agent alone, plus
`--ak alone=continue`:

    --ak agent=mini-swe-agent --ak services=none --ak alone=continue --ak reply_reserve_seconds=10
    --ak spend_cap_usd=2 --ak worker_spend_cap_usd=2 --ak monitor_spend_cap_usd=1
    --ak worker_max_calls=1000 --ak monitor_max_calls=400 --ak max_commands=2000

The agent (mini-swe-agent 2.4.6, unchanged) works on the task as given. When
it finishes, it is run again in the environment it left, given the task as
given, until it has run 12 times (the supervised arm's limit) or the task's
time is used up. Nothing plans, monitors or checks its work.

## Tasks and runs

The go/no-go's 20 tuning tasks of Terminal-Bench 2.1, three runs each: 60
trials, GPT-5.6 Luna, on one Azure VM with 32 vCPUs, 16 trials at a time.

## Comparison

`scripts/study_report.py`, against the go/no-go's second-run arms
(`gng3-alone`, `gng3-taste`): taste against continue, and continue against
alone. Reported: task-runs solved, the per-task comparison, the paired
analysis, and cost. There is no decision rule: the result informs the
study's design.

How to read it: if the continue arm solves about as many task-runs as the
supervised arm, the supervised arm's gain comes from running the agent again,
and the study has to show what supervision, or undoing changes, adds beyond
that. If it solves about as many as the agent run once, the gain comes from
what supervision does.

The runs differ in time and in code: this code adds the rule that runs the
agent again and fuller records when a run is cut off at its end; neither
changes how the other two arms work.

## Budget

At most $10 of model spend (expected $3 to $5).

## Record

(Filled in after the run: the command, the report and what it shows.)
