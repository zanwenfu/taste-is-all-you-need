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

## Question 5, added 2026-11-02 before any critic ran

Can a critic judge a run from its git record, or does it need the live
environment? Both critics get the task and the same record: the plans, what each
worker was asked, the planner's assessments and the latest report. Both use
GPT-6 Luna at medium effort. The record critic gets nothing else and answers in
one reply. The live critic (`checker-v2`) can also run commands in a fresh
container holding the run's final files. The verifier's result on the run is the
truth; neither critic sees it.

Sample: runs with a usable final save whose control agreed, so their files show
what the verifier saw. Runs that depended on a live process are out by
construction, which favours the live critic less, not more. Reported: each
critic's false rejections and false acceptances, in both directions, and how
often the two disagree on the same run. Code: `scripts/critic_study.py`. A
six-run pilot comes first. The solved/failed split is known to the analysis
only, not to either critic.

## Reporting

Every clean result is reported, whatever its direction. Amendments are added
below with their date and reason, and none is made after looking at a test
task's outcome except to report a defect in the run itself.

## Record

**Amendments before any trial on these tasks (2026-11-01).**

1. Each kept save may reach 4 GiB and a trial's store 32 GiB
   (`KEPT_CHECKPOINT_BYTES`, `KEPT_STORE_BYTES`, and `--ak checkpoint_mib=4096`
   for the final save), because a partial save cannot be graded. In the
   rollback pilot 8 of 31 saves were partial at the usual bound.
2. Question 2's measure is named: a lost win is a run that ended failed and held
   an earlier saved state the verifier passes. It is reported separately from
   a mid-run break, which a verifier-level check of saved states cannot see.
3. Question 4 also reports the supervising layer's errors in both directions,
   from the same trials: certifications of work the verifier failed, and
   refusals of work it passed.
4. Power. The interval over 20 tasks in the go/no-go implies a standard
   deviation of about 0.29 for the per-task difference in share solved. Over 69
   tasks that gives a 95% interval of about +-7 points, and about 80% power for
   a gap near 10 points. A gap of 6 to 9 points may not be distinguishable.
   Question 1 is reported as an interval for this reason.
5. The pilot's control disagreed once: a run the verifier passed scored 0 when
   its final save was restored and graded (kv-store-grpc). The files came back
   exact; the two tests that failed need a server still running, and a save
   holds no processes. So a passing state is seen only where passing depends on
   files. Question 2 is therefore a lower bound, and how many runs the control
   shows depended on a live process is reported as a measure in its own right.
6. Pilot results, tuning tasks, one trial per arm: 2 of 3 solved in each arm,
   no exceptions or flags, 7 checkpoints with none failed or partial,
   $0.003 per attempt alone and $0.05 supervised. Grading the 10 saved states
   took 69 seconds. A first pilot found the planner refusing Bedrock's replies
   as of unknown cost; fixed before this one.

**Run (2026-11-02, 12:06 to 6:30 am ET, code ce02a58).** All 414 trials graded
but one; 1 trial raised (a verifier timeout on `mailman`). Model spend $18.78.

Question 1. Solved 81 of 207 alone and 94 of 207 supervised: a mean per-task
difference of +6.3 points, bootstrap 95% interval [0.0, +13.0], permutation
p = 0.093, better on 16 tasks, worse on 7, same on 46. $0.0030 and $0.0877 per
trial, about 29 times the cost. Leaving out the 4 tasks whose reference solution
failed (`caffe-cifar-10`, `protein-assembly`, `qemu-alpine-ssh`, `qemu-startup`,
which neither arm solved): +6.7 points, [0.0, +13.8].

Question 2. 1,294 saves graded, 3 could not be. Control over 204 final saves:
189 agree; 12 runs the verifier passed scored 0 from their files alone, on five
service tasks (`git-multibranch`, `hf-model-inference`, `pypi-server`,
`configure-git-webserver`, `mailman`); 2 failed runs scored 1 (`largest-eigenval`,
a verifier that does not repeat); 1 run had no reward. Of 113 runs failed at
their end, 3 held an earlier passing save. One of those also passes at its final
save, so it is the same non-repeating verifier. That leaves 2 lost wins, 1.8% of
failed runs (`model-extraction-relu-logits`, `sparql-university`), a lower bound.

Question 4. 1,095 checkpoints, none failed, 5 partial. 5 trials rolled back, 6
restores, all exact, the longest 8 seconds; 2 of the 5 trials ended solved. 21
of 103 goals closed as complete failed the verifier; 75 submissions were refused
in trials the verifier passed. 29 supervised trials and 6 alone are flagged, from
about 35 worker model calls whose outcome is unknown, each kept on record as such.

Question 3 and the hand check are not done yet.

**Kill test with the real model (2026-11-02).** The same 156 planner kill points,
the planner and its recovery running on GPT-6 Luna instead of a stand-in. Every
kill landed and every recovery exited cleanly. None was wrong: no point made more
than one paid call. 128 recovered and planned from a live reply; 28 refused to
repeat because they could not tell whether a call had gone out, 15 of those with
no call made. Refusing is safe but leaves the run for someone to restart.

**Kill test on delivery (2026-11-02).** 48 kill points inside the step that hands
a worker's result to the integration branch. None was wrong: 45 finished with the
worker's file whole and the target's other files intact, 3 refused to repeat, and
git found no damage. With the planner, checkpoint, terminal and worker paths,
that is 280 points on a stand-in model and 156 on the real one.

