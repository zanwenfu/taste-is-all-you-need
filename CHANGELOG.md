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

- **A lost reply is charged by the request it sent.** A request holds no more
  tokens than bytes, so its size bounds its cost. A run that lost one reply
  was charged its whole ceiling (about $60 in a benchmark trial); it is now
  charged what it is known to have spent, plus that one call's worst case
  (about $4 for a 270 KB request).
- A worker's report carries that account (`metadata.model_cost`), and the
  coordinator counts the known part as spent.

### Fixed

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
