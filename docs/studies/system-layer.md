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

**Question 5 (2026-11-02).** 189 runs, 187 with both verdicts. Accuracy against
the verifier: record critic 83.4%, live critic 84.0%, a difference of +0.6 points,
task-clustered 95% interval [-5.3, +6.1], exact paired test p = 1.0. False
rejections 15.0% and 11.3%, false acceptances 17.8% and 19.6%. All 189 saved
states restored exactly. The live critic cost $0.37 in all.

The planner's own decision to close, on the same runs, is right 84.5% of the time,
so neither critic improves on it overall. The record critic agrees with that
decision on 179 of 187 runs and the live critic on 160. Of 18 goals closed as
complete that failed the verifier, the record critic caught 2 and the live critic
7. Of 11 runs left open that passed, it recognised 1 and the live critic 6. The
live critic pays for this with 10 false acceptances among 89 open failed runs,
against 3, and 4 false rejections among 69 closed passed runs, against 2.

**Amendment 7, added after seeing Question 5.** The record critic reads the
planner's conclusions, so its agreement with the planner may be repetition. A
third critic, exploratory and decided after the results, gets the same runs with
the planner's reasons and assessments removed: what each worker was asked and the
latest report.

Result of Amendment 7. With the planner's reasons and assessments removed, the
record critic is right on 79.7% of the 187 runs, agrees with the planner's close
on 170, and catches 3 of the 18 false completions, against the live critic's 7
(paired exact p = 0.125, on small numbers). What the record holds does not show
which claims are false.

**Why runs failed (2026-11-02).** Two independent readers classed the 92 failed
runs that ended by a limit from their plan-by-plan history, and agreed on 86
(kappa 0.88): stuck on one blocker 55, blocked by a resource the agent names 17,
real progress cut off by the limit 14. They split on 6. With the 21 failed goals
closed as complete, of 113 failed runs: stuck 49%, closed wrongly 19%, blocked
15%, cut off 12%, split 5%. Blocked rests on the agent's claim. One reader calls
one run a regression, the other none; a regression that left no trace in the
plans would not show. Only the first and last five plans of a long run were read.

**Worker calls of unknown outcome.** 35 calls in 29 supervised trials and 6 in 6
alone trials stopped a worker as "model unavailable"; each was the last call in
its journal, and each is on record as unknown. The one error text that reached a
log is a Bedrock refusal on usage-policy grounds; the others left none. All 6
alone trials failed, since that arm has no one to restart a worker; 9 of the 25
supervised trials hit by this solved. Leaving out the 6 alone trials: +6.0 points,
[-0.2, +12.6].

**Amendment 8, added 2026-11-02, code ready, not yet run: a crash on real tasks.**
The kill tests stop at single steps on a drill goal. A trial can now take
`crash_after_seconds`: its owner sends SIGKILL to the goal's whole process tree
(coordinator, monitors and workers) at that moment, proves the tree drained, and
starts one new goal process from the same records, with the same deadline and
limits. The container, the terminal and its ledger belong to the owner and live
on. The owner's own death still only drains. A run that ends first is not killed
and says so. Each trial records whether the kill landed, how long the new process
took to start and to run its first command, the commands on each side of the kill,
the same command run again, requests paid twice, and calls cut off in flight.

A cut-off call went out and may be billed, and its answer is lost, so the
decision is asked again in a new request. That is counted as cut off, not as
repeated. A worker's interrupted command is ended by the terminal and receipted,
and is not resumed: work in that worker's head is gone and the planner plans again
from what the records and the task's files show. This measures recovery of a run,
not recovery of a worker's thought.

Tests cover the owner's path with stub services, and a real goal process killed
with a paid call in flight on a mocked wire, which then ran again and settled.
`scripts/check_azure_harbor.py --mode crashed-goal` is the proof on a real machine
with a real worker and mock model. It has not been run. No crash study runs before
it passes.

**Amendment 8, design, fixed 2026-11-02 before any real task was killed.** Each of
the 69 test tasks runs once under the supervised arm's settings and model, with
no retries, so that a run a kill left broken is counted and not run again. The
kill comes at a point drawn for that task from a fixed seed, between 20% and 80%
of the median length of its three earlier runs, and never before 30 seconds. A
run that ends before its kill point was not killed and is reported apart.

Reported: kills that landed; runs that resumed; failed recoveries, meaning a
landed kill with no new process or no grade; commands and paid calls repeated;
calls cut off; time to resume. The comparison is per task: the killed run's result
against the share solved in that task's three unkilled runs, with a bootstrap 95%
interval, over tasks whose kill landed and over all 69. If recovery works, the
interval covers zero and leans on no task in particular. The unkilled runs came
earlier on the same machine and model, not alongside, and the killed run has one
attempt on each task. A free check with a stand-in model on real Docker comes
first, then three tuning tasks with the real model.

**A recovery bug the kill test had missed (2026-11-02).** On a practice task a kill
landed during a planner call and the restarted goal died within seconds. The
restart found the world changed and wrote "not called, cost zero" over a call
whose receipt said it had gone out, and the goal's own audit stopped it. The kill
test had shown no fault here because nothing moved between its kill and its
restart, and in a real run the restarted runtime writes to control first. The test
now does the same (`planner-moved`). Before the fix 41 of its 156 kill points
crashed on recovery; after it none is wrong, on both planner paths. 138 plan again
and 18 refuse to repeat. At 17 points the interrupted decision is asked afresh, so
that call may be paid for twice; each is on the record. The earlier "156 points,
none wrong" held only for a world that stands still. The same fault fixed in
October's terms in `_world_movement` covered a reply already received, not a call
still out.

