# Changelog

Notable changes, newest first. Before 1.0 a minor version may change
behaviour and interfaces.

## Unreleased

### Added

- **Agents written by others, run unchanged as workers** (`taste.agents`,
  `--ak agent=mini-swe-agent`). mini-swe-agent keeps its own loop,
  configuration, tool and parsing; Taste supplies its model transport and its
  shell, records both for its monitor, stops it when the monitor judges it
  wrong or lost, and certifies its report like any worker's. A test runs its
  loop with its own model class and with Taste's and finds the same requests,
  messages and commands.
- **An agent run alone through the same machinery** (`--ak services=none`),
  the baseline arm of a comparison: a fixed rule plans (the task verbatim, one
  assignment, then the agent's own words as the reply), no monitor, no
  certification, one generation. Both arms are held to the same cap per trial.
  `scripts/go_no_go_report.py` compares two arms task by task, with spending
  by role. The first study's protocol is `docs/studies/go-no-go.md`.
- **Any admitted model in every role, the coordinator included**, and GPT-5.6
  Luna (`gpt-5.6-luna`, version 2026-07-09) among them, at its Azure Global
  Standard price.
- **Replaying a recorded decision** (`scripts/replay_decision.py`): rebuilds
  from a copy of a run's memory the coordinator's prompt for one planning
  request, the step judgement that stopped a worker, or a worker run's
  certification, with whichever code is on the path. Run with the code that decided, the coordinator's prompt
  matches the SHA-256 recorded when it was sent; with `--calls N` the same
  model is asked again. Every prompt change made after the first go/no-go run
  was checked this way before being kept.
- **A checkpoint of what a task changed in its container**
  (`DockerTerminalBackend.checkpoint`, `taste/brains/docker_checkpoint.py`),
  the first part of undoing an agent's work: the paths the container changed
  against its image, as Docker lists them; the added paths (a top-most added
  directory whole) and changed files copied out into one tar stored under its
  SHA-256; the deleted paths listed; the kernel's trees and the container's
  mounts left out and listed; a size cap past which the checkpoint is marked
  partial. Taken only between commands. A container that changed nothing
  (Docker lists its changes as null) has an empty checkpoint. Checked against
  a real daemon by `scripts/check_docker_checkpoint.py` (contents, modes, a
  symbolic link, a changed and a deleted image file).
- **A restore of a checkpoint** (`DockerTerminalBackend.restore`): everything
  the task added since is removed, the image's own versions of files changed
  or deleted since come back from a helper container made from the same image
  (never started, removed afterwards), the checkpoint's files are put back and
  its deletions made again; then the container's changes are compared with the
  checkpoint's, and every difference is listed in the receipt. Running
  processes are not restored. Refused for a checkpoint of another image or
  container, or one whose tar no longer matches its checksum. Against a real
  daemon (`scripts/check_docker_checkpoint.py`), a container broken after its
  checkpoint was restored in 2.3 seconds with every file as it had been.
- **Checkpoints and restores at the coordinator's request**
  (`TerminalBroker.checkpoint` and `restore`, `TerminalIssuerClient.checkpoint`
  and `restore`). The controller runs them between commands, holding the
  terminal as a command does, so no command runs and nothing is handed to
  grading meanwhile. Each is recorded in the terminal ledger before and after,
  at most once per ID: asking again reads the record, a restore never runs
  twice, and a caller that leaves does not stop one halfway. A failure is
  recorded with the error's class, never its text, and commands go on. Only
  the coordinator's private credential may ask; a worker's is refused like a
  forged one. A checkpoint copies at most 512 MiB and a trial's store holds
  at most 4 GiB. After a controller restart, a restore that was begun and
  never recorded fences the environment.
- **A rollback report** (`scripts/rollback_report.py`): per trial of a
  Harbor job, every checkpoint and restore from the controller's ledger (size
  and counts, exact or not, how long, or the error's class) and every
  rollback with the checkpoint and reason the plan gave; per job, totals,
  restore times and the checkpoint stores' size on disk. Read-only.
- **Rollback decided by the coordinator** (`--ak rollback=on`,
  `taste/brains/environment_records.py`). The coordinator's runtime takes a
  checkpoint of the task's files before the first worker starts and after
  worker runs end, before each new plan, and records each on the control
  branch. The planner is shown the checkpoints (what the files then held
  against the image, by counts and first paths) and may name one in a
  proposal (`rollback_to`, with its evidence in `rollback_reason`); a name not
  listed, or a rollback without evidence, refuses the proposal. The files are
  restored once per plan: before its workers start, or before the goal ends
  when the plan is complete or closing. A restore that fails starts no worker
  and asks for a new plan. Reports and records of the undone runs stay in
  memory as history. With rollback off (the default) prompts, plans and
  policies are byte for byte as before. Against a real container
  (`scripts/check_rollback_wiring.py`, no model), the coordinator's
  checkpoints and restores through the controller's socket returned the files
  exactly, in 1.6 and 2.4 seconds, and asking again read the record. A
  trial's checkpoint tars are deleted once its environment is sealed for
  grading or stopped, when no restore can follow; each checkpoint's manifest
  and ledger record stay (in the rollback pilot one trial's after-run
  checkpoints reached 0.5-0.6 GB each). A partial checkpoint (past its size
  cap) is never offered and never restored; a restore is not started without
  time for its size; a checkpoint or restore whose coordinator gave up while
  it waited is dropped; a rollback the last plan named is made before the
  closing reply; and "", "none" or "CP2 " are read as null or cp2.
- **An export of a study's trial records** (`scripts/export_records.py`):
  each trial's Harbor result and verifier output and Taste's settled record,
  nothing else, with a manifest of SHA-256 checksums that `--verify` checks;
  refused if a copied file holds the value of a named secret. The report
  scripts run on the export as on the originals: the first go/no-go run's
  export (602 files, 176 MB, no Azure key in any file) recomputes its report
  ([#39](https://github.com/zanwenfu/taste-is-all-you-need/issues/39)).
- **A check of what passing trials fetched online** (`scripts/online_check.py`):
  every command a passing trial's agent ran, read from its settled record;
  each fetch from the internet (a URL, or curl, wget, git clone and the like
  run as a command; not package sources or the container's own addresses)
  classed as named by the task, solution-like (the benchmark, its
  repositories, a solution, or the task itself) or other, for review. On the
  first go/no-go run's 41 passing trials: none solution-like
  ([#37](https://github.com/zanwenfu/taste-is-all-you-need/issues/37)).
- **The study's analysis** (`scripts/study_report.py`): registered
  comparisons between arms, paired by task: per-task differences in the share
  of runs solved, a bootstrap interval over tasks, a sign-flip permutation
  test (exact up to 20 differing tasks) with Holm's adjustment over the
  comparisons, the exact sign test, and cost per trial and per solved task.
  Seeded, so the same records give the same report; checked on synthetic
  records with known answers
  ([#35](https://github.com/zanwenfu/taste-is-all-you-need/issues/35)).
- **The continue control** (`--ak services=none --ak alone=continue`): the
  agent alone, run again in the environment its last run left, given the
  task as given each time, until the supervised arm's generation bound or the
  task's time runs out. A gain from supervision might come from running the
  agent more than once; this arm has the attempts without the supervision
  ([#34](https://github.com/zanwenfu/taste-is-all-you-need/issues/34)).
- **A fixed split of Terminal-Bench 2.1 into tuning and test tasks**
  (`data/splits/terminal-bench-2-1.json`, rule in
  `taste/benchmarks/task_split.py`), and `infra/azure/run-terminal-bench.sh`,
  which runs one part and refuses test tasks unless a registered study names
  the split.
- **A time ceiling for each model request** (`request_seconds`, 300 in
  benchmark trials), inside the goal's deadline. A request the provider never
  answers ends there instead of holding its worker, or the coordinator, until
  the goal ends.
- **A spend cap the goal is held to** (goal metadata `spend_cap_usd`). Once a
  goal's known spending reaches it, no new plan is asked for and the goal
  closes with its reply (stop reason `spend_cap`). Benchmark trials already
  disclosed a cap of $15; nothing enforced it.
- On the Azure route, a request that could not be connected is sent again,
  as a rate-limit refusal already was: nothing was sent, so nothing was
  billed. A connection has 30 seconds to open.

### Changed

- **The planner writes only its decisions.** What the request and the policy
  fix (echoes of the request, schema names, an assignment's generation, base,
  model, routing and caps, and no inputs for a hosted agent) is filled by the
  harness and recorded with the plan as `filled_by_harness`; the plan stays
  bound to exactly what the model wrote. Measured on GPT-5.6 Luna, such slips
  were refused plans, and four ended a goal. Plans are also asked for as one
  JSON object, which the Azure Responses API enforces.
- A run whose spending is settled but which stopped some other way (its
  planner failing, a runtime error) is flagged `stopped:<reason>`, not
  `accounting_unsettled`.
- **A lost reply is charged by the request it sent.** A request holds no more
  tokens than bytes, so its size bounds its cost. A run that lost one reply
  was charged its whole ceiling (about $60 in a benchmark trial); it is now
  charged what it is known to have spent, plus that one call's worst case
  (about $4 for a 270 KB request).
- A worker's report carries that account (`metadata.model_cost`), and the
  coordinator counts the known part as spent.

### Fixed

- **A trial is graded even when a command was still being ended at its
  end.** A worker stopped at the wall clock leaves its last command to the
  controller, which ends it alone and confirms its exit within 20 seconds;
  sealing the environment for grading in that window was refused, and the
  trial went ungraded (2 of 60 supervised trials in the go/no-go's second
  run). The hand-off now waits up to 30 seconds for the terminal to fall idle
  (`TerminalBroker.wait_idle`), through the benchmark's own cancellation as
  settlement does, and interrupts nothing
  ([#49](https://github.com/zanwenfu/taste-is-all-you-need/issues/49)).
- **Work done in a task's container is judged where it is.** The coordinator
  assessed certified work "not met" because memory held only the run's
  report, and then wrote contracts requiring the task's files in memory, which
  no worker can satisfy and the certifier enforced. The coordinator is now
  told that a container's files never appear in memory and that the evidence
  about them is the commands workers ran there (`task_environment`), and both
  monitor prompts say a file's absence from the state's manifest is not
  evidence that it is missing
  ([#40](https://github.com/zanwenfu/taste-is-all-you-need/issues/40)).
- **A documented failure no longer meets a goal.** A goal was closed as
  complete on evidence of "17 passes and one documented failure", and the
  benchmark's verifier failed it: the rule that lets a worker's honest finding
  satisfy its contract was applied to the goal's own criteria. The coordinator
  is now told that a failing test or check, an error, unfinished work or a
  documented limitation means not met (`assessment_standard`,
  [#44](https://github.com/zanwenfu/taste-is-all-you-need/issues/44)).
- **A hosted agent is not asked to write a report.** The harness writes its
  report; contracts that asked the agent for one turned the agent's own
  reports into stops and refusals of otherwise right work
  (`hosted_contracts`,
  [#41](https://github.com/zanwenfu/taste-is-all-you-need/issues/41)).
- **The step monitor is shown the work before the batch it judges.** Shown
  only a run's last batch, which often holds just the submission, it reported
  36 times in the first go/no-go run that an agent "submitted after only a
  completion echo"; every one of those runs had recorded 3 to 14 commands.
  Each step observation now carries a mechanical record of the commands run
  before the batch (`earlier_in_this_run`) and whether the worker is still
  running
  ([#43](https://github.com/zanwenfu/taste-is-all-you-need/issues/43)).
- **A hosted agent's run cut off at its end leaves a whole record.** A run
  stopped while a model call was in flight left the reply, paid and
  journaled, out of the record; a command that lost its terminal at a trial's
  end was left without a result. In the first go/no-go run four of 60
  supervised trials were flagged `evidence_incomplete` for this. A cut-off
  reply is now recorded (marked `cut_off`, since the agent never received
  it), a call whose outcome is unknown is given up and charged its worst
  case, a command is recorded as ended (`cancelled` or
  `terminal_unavailable`, effects unknown), and a cancelled run waits for its
  call in flight to settle
  ([#48](https://github.com/zanwenfu/taste-is-all-you-need/issues/48)).
- **The certifier names earlier findings by label.** It had to copy every
  earlier monitor finding's 64-character id into one of two lists, and a
  mis-copied id made certification fail closed: two of ten refusals in a
  pilot. Findings are now shown as F1, F2, ... and the labels mapped back to
  their ids before the partition is checked
  ([#46](https://github.com/zanwenfu/taste-is-all-you-need/issues/46)).
- **A defect still being fixed is not a reason to stop an agent.** About half
  of the agents stopped mid-run in the first go/no-go run were developing,
  on a failing check they could still fix ("fails a basic sampling call", in
  a trial later solved). The step prompt now says that while the worker is
  running such a defect is drifting at most, and what wrong is for: working
  against the contract
  ([#45](https://github.com/zanwenfu/taste-is-all-you-need/issues/45)).
- **Only refused plans in a row end a goal.** The limit counted refusals over
  the whole goal, so a long goal ended although its planner recovered every
  time. A key at a proposal's top level that is not part of a proposal (an
  echoed prompt section or plan id) is dropped and recorded with the plan, a
  hosted agent's assignment is given its report as its one output (plans had
  declared the task's own file beside it), and a criterion id copied with a
  slip is written, when the slip is unambiguous
  ([#42](https://github.com/zanwenfu/taste-is-all-you-need/issues/42)).
- **A lost reply no longer ends the run.** A worker whose request got no
  reply (a timeout, a dropped connection, the provider's own passing trouble)
  ended its run, and another worker began the assignment again. The journal
  now gives the lost call up, charges it the most the request it sent can have
  cost (measured by the service where the request continued an answered one),
  and the worker asks the same question as a new call, at most twice. Its
  monitor does the same, once. A lost final certification is no longer
  recorded as a refusal.
- **A killed worker leaves a record.** A worker killed after its grace period
  wrote nothing, and the coordinator closed knowing nothing of it. What it ran
  is now read from its own recorded turns and given to the planner with the
  trigger that reports the run, and with the closing trigger: each command,
  its exit and the last line it printed.
- **The certifier can read a worker's report.** An output artifact over 8 KB
  was shown to the certifier as omitted, so a run whose criteria spoke of its
  report was refused and a second worker sent to write a shorter one (four of
  twelve pilot trials). With room in its request limit, the certifier is now
  shown artifacts of up to the 64 KiB a worker may write.

## 0.2.0 (2026-10-02)

The first tagged release. Research software, alpha: see "Status" in the
README for what has and has not been shown.

### Added

- **Memory layer** (`taste/memstore/`): a session's memory as one git
  repository, with leased branches, atomic checkpoints, verdicts, inboxes,
  typed merges and rollback as an append.
- **Multi-process runtime** (`taste/brains/`): a central brain (planner,
  runtime, supervisor), one operating-system process per worker, an
  observe-only monitor per worker, and delivery of certified work into an
  integration branch.
- **Two worker harnesses**: Claude Agent SDK workers on their own git
  worktrees, and Azure OpenAI Responses workers with a journaled record of
  every model call.
- **Terminal broker** for a shared task container: one command at a time,
  every command on record, and a timeout that ends only the command.
- **Harbor benchmark agent** (`taste.benchmarks.harbor_agent:TasteAgent`),
  with a launcher, a host check and a run guide under `infra/azure/`.
- **Closing reply**: a goal that runs out of time or budget still ends with
  a reply that says what was done and what was not.
- Architecture figure and document, `CITATION.cff`, `CONTRIBUTING.md`.

### Changed

- **License: Apache-2.0** from this release, with a `NOTICE`. Earlier
  revisions of the repository were published under the MIT License.
- README rewritten for readers; the harness reference notes moved to
  `docs/azure-harness.md`.
- Paper drafts moved to the branch `archive/paper-2026`.

### Known limits

Tracked as [issues](https://github.com/zanwenfu/taste-is-all-you-need/issues).
The largest: no full benchmark run yet, one worker at a time on a shared task
container, no command line for the multi-process runtime, and a benchmark
agent that needs a Linux host with root, systemd and local Docker.

## 0.1.0 (not tagged)

The single-process kernel: a planner, a worker and a monitor in one loop, a
commit after every step, rollback on a failed check, parallel waves on git
worktrees, fault recovery, checkpoint journal, tool guardrails, and the
`taste` command with its HTML dashboard.
