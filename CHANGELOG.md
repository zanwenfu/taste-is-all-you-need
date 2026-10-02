# Changelog

Notable changes, newest first. Before 1.0 a minor version may change
behaviour and interfaces.

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
