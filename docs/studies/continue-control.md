# Continue control: does trying again explain the gain?

Registered on 2026-10-05, before any trial of this arm ran
([issue 34](https://github.com/zanwenfu/taste-is-all-you-need/issues/34)).
The code and settings are fixed at the commit tagged `continue-control-2`
(see the amendment below; `continue-control-1` was the first registration).

## Question

In the go/no-go's second run (`docs/studies/go-no-go.md`), the agent under
Taste solved 25 of 60 task-runs and the agent alone 20 of 60. Taste runs the
agent again when its work is not done. Does running the agent again, with no
planning, monitoring or checking, do as well?

## Arm

`continue`: the go/no-go's settings for the agent alone, plus
`--ak alone=continue`:

    --ak agent=mini-swe-agent --ak services=none --ak alone=continue
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

**First attempt, at `continue-control-1` (2026-10-05, 00:17 to 00:50 UTC),
stopped and not reported.** Two faults, neither the arm's. At 00:44 a test
suite run on the same VM filled its temporary space (`/tmp`, held in
memory), and about half of the trials then failed with "no space left on
device". Before that, seven trials had ended with the goal's process killed
at the task's time limit: this arm keeps the agent working until the time
is up, and the 10-second closing reserve copied from the agent-alone
settings left no time to stop and record the run still going. The trials'
records are kept on the VM under `/root/tb/jobs/cc1-continue`.

**Amendment (2026-10-05, before the second attempt): code at
`continue-control-2`.** The continue arm keeps the supervised arm's closing
reserve (the default, 210 seconds) instead of 10 seconds: a run still going
at the end is then stopped and recorded within the task's time, and the two
arms have the same time to work, which the comparison with the supervised
arm needs. Nothing else runs on the VM during the run.

**Run, at `continue-control-2` (2026-10-05, 00:52 to 01:59 UTC).** One Azure
VM with 32 vCPUs, 16 trials at a time, nothing else running on it. From the
tagged code:

    MODEL=azure/gpt-5.6-luna infra/azure/run-terminal-bench.sh cc2-continue tuning -k 3 -n 16 \
        --ak agent=mini-swe-agent --ak services=none --ak alone=continue \
        --ak spend_cap_usd=2 --ak worker_spend_cap_usd=2 --ak monitor_spend_cap_usd=1 \
        --ak worker_max_calls=1000 --ak monitor_max_calls=400 --ak max_commands=2000
    python scripts/go_no_go_report.py --arm continue=/root/tb/jobs/cc2-continue
    python scripts/study_report.py --arm alone=/root/tb/jobs/gng3-alone \
        --arm continue=/root/tb/jobs/cc2-continue --arm taste=/root/tb/jobs/gng3-taste \
        --compare taste:continue --compare continue:alone --compare taste:alone

| Arm | Solved | Trials | $/trial | $/solved |
| --- | --- | --- | --- | --- |
| alone (go/no-go, second run) | 20 | 60 | 0.006437 | 0.019312 |
| continue | 19 | 60 | 0.042830 | 0.135252 |
| taste (go/no-go, second run) | 25 | 60 | 0.194744 | 0.467385 |

| Comparison | Tasks | Mean difference | Bootstrap 95% | Better / worse / same | Permutation p | Holm p | Sign test p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| taste vs continue | 20 | 0.100 | [-0.050, 0.267] | 6 / 3 / 11 | 0.3438 | 1 | 0.5078 |
| continue vs alone | 20 | -0.017 | [-0.150, 0.117] | 3 / 3 / 14 | 1 | 1 | 1 |
| taste vs alone | 20 | 0.083 | [-0.033, 0.217] | 5 / 3 / 12 | 0.3438 | 1 | 0.7266 |

All 60 trials were graded; none raised an exception or was flagged. 44
ended at the task's time and 16 after 12 runs; the agent ran 487 times in
all (482 ended by its own submission, 1 by repeated format errors, 4 with
no exit of their own: cut off at the task's end). Model spend $2.57.

Per task, the continue arm did better than the agent run once on 3 tasks
(adaptive-rejection-sampler 1 of 3 against 0, mcmc-sampling-stan 2 against 0,
sanitize-git-repo 2 against 1) and worse on 3 (count-dataset-tokens 0 against
1, portfolio-optimization 0 against 3, sqlite-with-gcov 0 against 1). The
three portfolio-optimization trials failed only the task's speed test: the
final code was 1.16 and 1.197 times as fast as the baseline where 1.2 is
required, and 0.97 times in the third. Whether later runs made a correct
solution slower, or the busier VM slowed the test, these records do not
settle; it is a question for the rollback pilot.

**What it shows.** Running the agent again until its time is up, with nothing
planning, monitoring or checking, solved as many task-runs as running it once
(19 against 20), at seven times the cost. The supervised arm's 25 is above
both, by margins that 20 tasks cannot tell from none. So the supervised arm's
gain, whatever its size, does not come from running the agent again; the
study has to show what supervision, and rolling back, add.
