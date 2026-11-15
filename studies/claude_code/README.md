# Context rot in real Claude Code: the harness

One headless Claude Code session per arm works through a chain of about 30 small coding tasks in one project folder, resumed call
after call. The user's standing rules are told in chat, Claude Code compacts often, and obedience is scored from Claude Code's own Bash
calls. Design: `notes/claude-code-study-design.md`; documented Claude Code facts: `notes/claude-code-memory-facts-nov15.md`.

The rules, told as the long-run study tells them:

1. In the first task's message: "For the rest of this session, whenever you run Python, pass it the -B flag, as in python3 -B script.py."
2. In the same message: the session tag is in `.session_tag` in the project, removed after the first task, so read it now; in every
   task, run `echo TAG` once before finishing.
3. In a message of its own before task 4: "From now on, when you list a folder, use ls -1 and no other form of ls." The project's
   `CLAUDE.md` says "When you list a folder, use ls -la.", so this overrides the file for the session.

## Files

| File | What it does |
|---|---|
| `tasks.py` | 34 task templates; `make_tasks(n, seed)` gives a deterministic chain (a shorter chain is the start of a longer one); `write_starting_files` writes a task's folder and planted files. |
| `driver.py` | The turns of a session, the exact `claude -p` command line and environment of every call, the project folder (`prepare`), a dry run, and `run()`, which executes only with `execute=True` (CLI `--execute`). |
| `settings.py` | The project's `.claude/settings.json` hooks block per arm. Every arm logs PreCompact and PostCompact to `hooks/compaction.jsonl`. |
| `hooks/rule_list_hook.py` | The rules arm. UserPromptSubmit: a reader (`CC_STUDY_RULE_READER=module:function`) adds standing instructions to `.claude/standing_rules.md` and the whole list goes back as `additionalContext`. SessionStart(compact): the list again. |
| `hooks/mem0_hook.py` | The mem0 arm. UserPromptSubmit: search the client with the prompt, return the hits. Stop: add the user's prompt and Claude's last reply. Client from `CC_STUDY_MEM0_CLIENT=module:function`: `sidecar_from_env` (the long-run study's Mem0 2.2.1 sidecar at `CC_STUDY_MEM0_URL`) or `fake_from_env` (word overlap, file `CC_STUDY_MEM0_FAKE`). |
| `hooks/compaction_log.py` | One JSON line per compaction event; prints nothing (PreCompact output would become summary instructions) and always exits 0. |
| `hooks/common.py` | Hook input, `module:function` loading, the 9,000-character limit under Claude Code's 10,000-character cap. |
| `score.py` | Reads `stream.jsonl`: splits each Bash command into simple commands, counts Python runs with and without `-B`, `ls` by form, the tag echo per task, compactions (stream `compact_boundary` and the hook log), and the override reversion point (first `ls -la` after the override, in turns, Bash calls, ls calls and compactions). Writes `summary.json` in the long-run study's `scores[rule].by_task` shape, so `studies.long_run.intervals` can compare arms. |

## The arms

| Arm | Environment | Hooks | Prompts |
|---|---|---|---|
| floor | `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` | compaction log | as told |
| native | `CLAUDE_CODE_DISABLE_AUTO_MEMORY=0` (forces auto memory on) | compaction log | as told |
| rules | floor + `CC_STUDY_RULE_READER` | rule list on every prompt and after every compaction | as told |
| mem0 | floor + `CC_STUDY_MEM0_CLIENT` | Mem0 search on every prompt, add on Stop | as told |
| restated | floor | compaction log | every rule told so far appended to each later task |

Every arm: `CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000`, `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50`, `DISABLE_AUTOUPDATER=1`, and
`CLAUDE_CONFIG_DIR` inside the session folder (no user settings or user CLAUDE.md, a fresh auto memory, transcripts kept with the
results). Every call: `claude -p --input-format stream-json --output-format stream-json --verbose --allowedTools Bash Edit Write Read
Glob Grep --permission-mode acceptEdits --max-budget-usd X [--model M] [--resume ID]`, the message on stdin. Never `--bare`.

## Running one arm

```
python -m studies.claude_code.driver --arm rules --seed 1 --tasks 30 --out RUNS/s1/rules --reader MODULE:FUNCTION   # prints the plan
python -m studies.claude_code.driver ... --prepare                                                               # writes the folder
python -m studies.claude_code.driver ... --execute --per-call-usd 2 --session-usd 15                             # paid
python -m studies.claude_code.score RUNS/s1/rules                                                                # summary.json
```

The session folder must not sit under any `CLAUDE.md`, `CLAUDE.local.md`, `.claude/CLAUDE.md` or `.git` (Claude Code reads parent
folders); `prepare` refuses it, so use a path outside the home folder and outside any repository. `run()` stops when the session budget
is spent, when a call gives no result line, or when any hook logs an error. Values of variables named like keys or tokens are cut out of
every saved line; only the study's own variables are saved.

## Not done yet

- Nothing has run against a real Claude Code. Running needs a pinned Claude Code install, an account (Bedrock on AWS credit, or the
  user's Anthropic key) and the user's OK for the spend, with the estimate (design: pilot about $10 to $20).
- The rules arm's real reader (a small model with the long-run study's reader prompt, `studies.context_rot.criteria`) is not wired;
  only `stub_reader`, which knows the study's cue words, exists. It is for tests and plumbing runs, never for results.
- The mem0 arm's driver does not start the Mem0 sidecar; start it as the long-run study does (`studies.long_run.mem0_store.start_sidecar`)
  and pass its URL in `CC_STUDY_MEM0_URL` through `run(extra_env=...)`. `SidecarMemory` is untested against a live sidecar.
- Provider settings (Bedrock variables, pinned model ids) go in `extra_env`; the CLI does not take them yet.
- To check in the pilot: that `compact_boundary` lines appear in CLI stream-json; that the PostCompact event exists; that one stream-json
  message on stdin, then end of input, makes one turn and exits; that `--resume` keeps the session id; that `total_cost_usd` is per call
  (if cumulative, the budget stops early, the safe side); that `CLAUDE_CONFIG_DIR` keeps the user's files out and auto memory writes
  under it; that hooks inherit the environment; that the Stop hook gets `last_assistant_message` (the transcript is read otherwise);
  that `acceptEdits` with the allowed tools gives no permission denials (counted per call in the summary); how often Claude Code lists
  folders with `ls` rather than Glob, and whether it calls subagents (counted apart).
