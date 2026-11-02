# What a git systems layer makes reliable, on tasks it was never tuned on

Registered on 2026-11-01, before any trial on these tasks, in
`data/studies/system-layer-1.json`. Every number below comes from
`scripts/study_report.py`, `scripts/rollback_report.py` and
`scripts/grade_saved_states.py`, run over the job folders, which are kept.

## Why

The go/no-go and continue-control studies used the 20 tuning tasks, on which
defects had just been found and fixed, and their job folders were not kept.
They left three questions open. Supervision solved 25 of 60 task-runs against
20 of 60 alone, at about 30 times the cost, a gap 20 tasks cannot tell from
none. The reading of 128 supervised records found few regressions, but those
records show only what the planner saw. And rollback, offered in 8 trials, was
never chosen, so whether it had anything to act on is unknown.

## Questions

1. On tasks never used in development, does the supervised arm solve more
   task-runs than the agent alone, and at what cost? Reported as an interval
   whichever way it falls. An interval that includes zero is reported as a gap
   these tasks cannot tell from none, never as equivalence.
2. Of the supervised runs the verifier failed at their end, how many held a
   saved state earlier in the run that the verifier passes? This measures what
   a rollback could have recovered, from the task's own check rather than from
   the planner's records.
3. How did the failed supervised runs fail? Read from the records, with the
   classes of the earlier reading: stuck on one blocker, completed early,
   progress cut off by time, a defect found late, a regression the planner
   could see, unclear. Two independent readers, a sample of 20 checked by hand,
   and each reader's regressions compared with the answer to question 2.
4. Did the system's own guarantees hold? Trials ungraded or flagged, records
   unsettled, spend past the cap, saves that failed or came out partial,
   restores not exact, rollbacks chosen.

## Agent, model, tasks

- mini-swe-agent 2.4.6 with its own `mini` configuration, unchanged.
- GPT-6 Luna on Amazon Bedrock in every role: the agent at low reasoning
  effort, the coordinator at medium, the monitors at low. On this route
  reasoning is not carried between turns, and a plan is held to one JSON object
  through Converse's `outputConfig`.
- The 69 test tasks of Terminal-Bench 2.1 (`data/splits/terminal-bench-2-1.json`),
  each with its own published limits, in Harbor's local Docker environment on
  one machine. No trial has run on them before this study.

## Arms

Three trials per task per arm, 414 in all, both arms started together so that
neither gets a quieter machine.

| Arm | Settings |
| --- | --- |
| alone | `--ak agent=mini-swe-agent --ak services=none --ak reply_reserve_seconds=10` |
| supervised | `--ak agent=mini-swe-agent --ak rollback=on --ak keep_checkpoints=on --ak final_checkpoint=sl1` |

Both arms: `--ak spend_cap_usd=2 --ak worker_spend_cap_usd=2
--ak monitor_spend_cap_usd=1 --ak worker_max_calls=1000
--ak monitor_max_calls=400 --ak max_commands=2000`, as in the go/no-go.

## Grading saved states

Each supervised trial saves its files before its first worker and after each
worker run, and once more before its verifier. After the runs, each save is
restored in a fresh container of the task's image and the task's verifier runs
on it (`taste.benchmarks.saved_state_agent`). No model is called.

Control: a run's final save, graded this way, should get the reward the run
itself got. Disagreements are reported by task, and a task that disagrees is
left out of question 2 and counted. Saves that failed or came out partial are
counted and not graded.

## Validity of the tasks

Each test task's reference solution is run once through Harbor's oracle agent.
Tasks whose reference solution fails are reported. The primary comparison keeps
every task; a sensitivity analysis leaves them out.

## Budget and stopping

Paid from AWS credits. Estimate $40-60, stop at $80, model calls and machine
time together. If the first 40 trials project past the stop, the run stops and
the decision returns to the project owner. A six-trial pilot on tuning tasks
(`adaptive-rejection-sampler`, `kv-store-grpc`, `polyglot-c-py`, both arms),
including the saved-state grading, comes first. Any defect it shows is fixed
and recorded here before a test task runs. Nothing else runs on the machine
during the measurement.

## Reporting

Every clean result is reported, whatever its direction. Amendments are added
below with their date and reason, and none is made after looking at a test
task's outcome except to report a defect in the run itself.

## Record
