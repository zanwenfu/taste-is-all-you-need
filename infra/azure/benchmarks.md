# Running a Harbor benchmark with the Taste agent

`taste.benchmarks.harbor_agent:TasteAgent` runs one benchmark task as a goal in
Harbor's standard Docker environment. The task is run as published: its
container, user, working directory, network rule and time limit, with no
environment subclass, extra Compose file or override. The coordinator, its
workers and their monitors run on the host; only shell commands enter the task
container, one at a time.

This page is the procedure that has been run on the test VM. What has and has
not been validated is at the end.

## What the host needs

- Linux with cgroup v2 and systemd, Docker with the Compose and Buildx plugins,
  and nftables (Harbor's egress sidecar enforces a task's network rule).
- Harbor runs as **root**: the trial owner admits the task container through the
  local Docker socket and runs the goal in bounded systemd units.
- An unprivileged account with no supplementary groups (here `bugbash`) runs the
  goal's processes. It never gets the Docker socket.
- Two Python environments: Harbor's (`harbor==0.23.0`), and the worker's with
  this project's dependencies. Harbor's must be able to import `taste`
  (`PYTHONPATH`), from a checkout that root owns and the worker user can read.
- A root-only file (mode 600) with `AZURE_OPENAI_BASE_URL` and
  `AZURE_OPENAI_API_KEY`. The key reaches the goal as a private systemd
  credential. It never enters the task container, a prompt or a trajectory.

## Check the host

    sudo infra/azure/check-host.sh

It changes nothing and prints one line per requirement above, with what to do
about each one that is not met. `run-harbor.sh` runs it first and refuses to
start a job on a host that fails it.

## Run

    sudo infra/azure/run-harbor.sh <job> <tasks dir> [harbor run options]

    # errata-bench: Harbor must also find its agents and task digests. The
    # four tasks excluded are the ones its judge does not grade (see below).
    sudo EXTRA_PATH=/root/errata/errata-bench/src infra/azure/run-harbor.sh \
        errata-astra /root/errata/v1.0.2-dataset/harbor -k 3 -n 2 --max-retries 2 \
        -x Pavel401-BugViper-85 -x Whiteknight07-AiTutor-34 -x entireio-cli-253 -x entireio-cli-38

The script starts `harbor run -a taste.benchmarks.harbor_agent:TasteAgent
-m azure/gpt-6-astra` in one systemd unit (`taste-harbor-<job>`). Follow it
with `journalctl -u taste-harbor-<job> -f`. Results are under the jobs
directory, as for any Harbor agent.

The agent's own processes run on the host and call only the Azure OpenAI
endpoint; it has no web or search tool. Every command it runs in the task
container is subject to the task's network rule, which Harbor enforces there.

## Settings (`--ak name=value`)

These are the run's disclosed configuration; each trial records them in its
result metadata (`agent_result.metadata.taste.configuration`).

| Setting | Default | Meaning |
| --- | --- | --- |
| `-m azure/<model>` | `gpt-6-astra` | The trial's model, which the coordinator runs on and every other role too unless `worker_model` names another: `gpt-6-astra`, `gpt-6-sol` or `gpt-5.6-luna`. |
| `worker_model` | the trial's model | Served model for workers and monitors, one of the same three. |
| `agent` | empty | An agent written by others that every worker runs unchanged: `mini-swe-agent`. Empty runs Taste's own worker. See below. |
| `services` | `all` | `none` runs that agent alone: a fixed rule plans (the task verbatim, one assignment), no monitor judges it and nothing certifies it, one generation. The baseline arm, through the same machinery; the agent may spend the trial's whole cap. |
| `alone` | `once` | With `services=none`: `continue` runs the agent again, given the task as given each time, in the environment the last run left, until `max_generations` or the task's time runs out. The control for a gain that comes from more attempts rather than from supervision. |
| `rollback` | `off` | With `services=all`: `on` has the controller checkpoint the container's files before the first worker and after runs end, shows the coordinator those checkpoints, and restores the one a plan names (`rollback_to`) before that plan acts. Files only: running processes are not restored. |
| `worker_effort` | `low` | Worker reasoning effort: `low`, `medium`, `high`. Monitors use `low`. |
| `planner_effort` | `medium` | The coordinator's reasoning effort. Empty leaves the provider's default. Measured on gpt-6-astra, the level changes little: the model reasons briefly at every level. |
| `spend_cap_usd` | 15 | Known spending at which a trial takes on no more work and replies. A worker already running finishes first, so a trial can pass the cap by that worker's and its monitor's allowances and by the closing reply. |
| `worker_spend_cap_usd`, `monitor_spend_cap_usd` | 6, 2 | What one worker and its monitor may spend: neither is admitted another call once its spending has passed this. |
| `request_seconds` | 300 | Longest any one model request may wait for its reply. A request the service never answers ends there and is accounted as a lost reply. In 257 recorded calls the slowest took 40 seconds. |
| `max_assignments` | 1 | Assignments one plan may hold. One container and a serial terminal make one worker at a time the honest default. |
| `reply_reserve_seconds` | 210 | Held back from working time for the coordinator's closing reply. |
| `plan_seconds` | 90 | No new plan is started with less working time than this; the run closes instead. |
| `handoff_seconds` | 150 | Held back after that for settlement, the record and the handoff. |
| `command_seconds` | 600 | Longest timeout one command may ask for (default per command: 120). |
| `agent_timeout_sec` | from `task.toml` | Only if the task's published agent time must be overridden for a drill. |

The caps are what hold real spending. The admission budgets a trial also
records are much larger and are not what it is meant to spend. Before a call
is sent, its worst case is set aside, priced as if the request filled the
model's whole context (about $27 on gpt-6-astra). A role's budget is its cap
plus one such call, and a trial's is about $226 for a $15 cap. They show that
a trial cannot spend more than that whatever happens.

A call whose reply was lost has no known cost. It is charged the most the
request it sent can have cost: a request holds no more tokens than bytes, so
a 200 KB request is charged at most $3.40 for its input, not $27. That amount
stays set aside for the rest of the trial and is reported apart from what is
known to be spent.

## Agents written by others

With `--ak agent=mini-swe-agent` every worker is mini-swe-agent 2.4.6 with its
own `mini` configuration (the one Harbor runs it with), its own loop, prompts,
bash tool and parsing (`taste/agents/mini_swe_agent.py`). Taste supplies only
its model transport and its shell: each request goes through the worker's
journaled model session, each command to the task container through the
terminal broker, and both are recorded for the worker's monitor. A monitor
that judges the agent wrong or lost stops it at its next call. Its exit, final
message, submission and command record become the assignment's report.

mini-swe-agent is installed in the worker environment without its own
dependencies, which would replace the `openai` package this project needs:

    pip install -e '.[agents]'
    pip install --no-deps --require-hashes -r requirements-agents.txt

With `--ak services=none` the same agent runs alone, as the baseline of a
comparison: everything is the same (worker process, journal, broker, record)
except that a fixed rule plans and no monitor or certifier runs. The fixed
plan's calls are recorded like any plan, at no cost, as `taste-fixed-plan`.
`scripts/go_no_go_report.py` compares two such job directories task by task.

Differences from running it outside Taste, disclosed with results: each reply's
output tokens are capped (a budget needs a ceiling per call), its template
values leave out the host's environment variables, and output beyond the
terminal broker's retention limit is cut.

## What a trial leaves

- `<job>/<trial>/agent/trajectory.json`: the instruction, every worker step in
  the order it happened (one tool call per step, marked as a sidechain), and the
  coordinator's final reply. `extra.audit_flags` lists what this system's own
  audit lacks for that trial. An empty list means none.
- `/var/lib/taste-trials/<token>/`: the private record (root only): the complete
  settled trajectory with planner and monitor calls, the terminal ledger, every
  model call's journal and the goal's memory. `metadata.taste.trial` in the
  Harbor result names the token.

A trial is handed to the benchmark's verifier whatever the audit found. A run
that stops on time, its spend cap, its budget, the generation bound or planner
failures still ends with the coordinator's closing reply; one cancelled by the
benchmark's own time limit is sealed first and graded as a timeout.

## errata-bench

    # 1. unpaid: which trials can be graded, and why not the others
    cd /root/errata/errata-bench && /root/errata/grade-venv/bin/python scripts/grade_harbor.py \
        /root/errata/v1.0.2-dataset /root/errata/jobs/<job> --out /root/errata/runs/<job> \
        --admission /root/errata/v1.0.2-dataset/admission/gpt-6-astra --rows-only

The current release is errata-bench's code at tag `v1.0.4` with its tasks at
dataset version `v1.0.2` (checked 2026-10-02). Its `main` branch is ahead, for
a next version of the tasks that is not released, and does not run these.
The official judge is admitted on 51 of the 55 tasks and only answers on
those are graded, so a full run is 153 trials. The other four are
`Pavel401-BugViper-85`, `Whiteknight07-AiTutor-34`, `entireio-cli-253` and
`entireio-cli-38`.

**A run is on hold.** On 2026-10-02 errata-bench reported that a v1 task can
start from a stale or wrong commit
([its issue 18](https://github.com/zanwenfu/errata-bench/issues/18)): 27 of
the 55 tasks have a confirmed start. Two of the tasks used below are among
those it names as wrong, `135yshr-savanna-vet-go-28` and `hutusi-amytis-18`.
The trials recorded on this page show that this agent runs, settles and is
accepted for grading. They are not results, and a full run waits for
errata-bench's next task release.

Grading itself is paid and runs in errata-bench's own environment with the
judge's key (see errata-bench's running guide). Taste's reply is written by
gpt-6-astra, which is also errata-bench's judge, so its self-grading check
refuses unless `ERRATA_ALLOW_SELF_GRADING=1` is set; results graded that way
must be reported as self-graded.

The dataset on the VM was copied from the local v1.0.2 release folder and
every file except `README.md` verified against the published `SHA256SUMS`.

## Terminal-Bench 2.1

    # once: the 89 tasks, from Harbor's registry
    sudo /root/taste-harbor-20260927/venv/bin/harbor download \
        terminal-bench/terminal-bench-2-1 -o /root/tb

    # tuning tasks: any time
    sudo infra/azure/run-terminal-bench.sh <job> tuning -k 1 -n 2
    # test tasks: only for a registered study
    sudo infra/azure/run-terminal-bench.sh <job> test <study> -k 3 -n 2

Settings may be tuned only on tasks whose results are not reported, so the
tasks are split once, before any run, by the rule in
`taste/benchmarks/task_split.py`: 20 tuning tasks stratified by difficulty,
taken in the order of a salted hash, no category beyond its share. The split
is `data/splits/terminal-bench-2-1.json` (salt `taste-tb21-split-2026-10-04`);
it pins the dataset by a digest of every file and is never rewritten.
`run-terminal-bench.sh` copies one part's tasks into a new directory and runs
them. It refuses a dataset that differs from the pinned one, and test tasks
unless the study is registered in `data/studies/<study>.json` with the SHA-256
of the split record.

## Validated so far (2026-10-01, test VM, gpt-6-astra in every role)

Fifteen trials of four errata-bench v1.0.2 tasks, one to five at a time.
errata-bench's unpaid admission check (`--rows-only`) counts all fifteen as
gradable; the fourteen run with the official time limit are counted official.

| What was exercised | Result |
| --- | --- |
| Completion, small instruction (2 to 10 KB), three tasks | 80 to 144 s, $0.89 to $1.72, 2 plans, no audit flags |
| Completion, large instruction (117 KB), twice | 263 and 275 s, $4.27 and $4.46, 2 plans, 13 and 14 commands, no audit flags |
| Two attempts of one task at once; five trials at once | no interference |
| The benchmark's own time limit cancels the agent (drill) | sealed within 5 s, `AgentTimeoutError`, commands recorded in order, graded as no answer |
| Working time runs out (drill) | worker stops in good order, exact cost, closing reply names what was run and what was not verified |
| A command that ignores TERM and leaves a detached child (`scripts/check_docker_terminal.py`, no model) | ended alone in 1.1 s, container and terminal still usable |

The first run of each kind found a defect, and each was fixed and run again;
the commit messages from 2026-10-01 describe them one by one. The largest was
a certifier that could not read a long run's evidence: the same 117 KB task
cost $18.58 and took 17 minutes before that fix.

### Pilot (2026-10-02)

Twelve further tasks, from twelve other repositories, one attempt each and
two at a time. All twelve completed, and errata-bench's unpaid check counts
all twelve as gradable and official.

| | |
| --- | --- |
| Time per trial | 76 to 714 s, median 273 s, of the 1,440 s of working time |
| Cost per trial | $0.95 to $8.70, median $2.67, mean $3.67; $44.08 in all |
| Where the money went | coordinator 29%, workers 41%, monitors 30% |
| Largest worker request | 382 KB |
| Audit flags | none |

The pilot found one defect. Four of the twelve trials needed a second worker
only because the certifier could not read the first worker's evidence report:
a report over 8 KB reached it as "omitted", and the criteria about what the
report records could not be certified. The certifier is now shown a report of
up to the 64 KiB a worker may write. Two of those four tasks were run again:

| Task | Before | After |
| --- | --- | --- |
| `blittle-pressy-158` | two workers, 714 s, $8.19 | one worker, 465 s, $5.73 |
| `melagiri-code-insights-53` | two workers, 446 s, $4.19 | one worker, 266 s, $2.48 |

Three drills on the real service checked the limits added the same day: an
ordinary trial (169 s, $1.76); a spend cap of $0.50 (stopped at $1.21 known,
closing reply written, gradable); and a request ceiling of 15 seconds (four
plans cut off at 15 s each, none sent again, the trial settled and sealed in
64 s with no reply).

### What a full run would cost

Most errata-bench instructions are large: the median is 75 KB and 32 of the
55 tasks exceed 64 KB. Cost follows instruction size. At the pilot's costs,
with the certifier fix, a full run of 153 trials is in the region of $550 to
$750 for the agent. Its spend caps hold a trial to $23 and three calls: $15,
then a worker and its monitor that were already running ($6 and $2), and the
one call each of them and the closing reply may have in flight. The dearest
of 774 calls recorded cost $1.17, so the run is bounded near $4,100. Paid
grading is separate.

Not validated: a full three-attempt run, paid grading, more than five trials
at once on a larger machine, Terminal-Bench's separate verifier, and any
environment other than local Docker.

Older fixture scripts under `scripts/check_*.py`, other than
`check_docker_terminal.py`, predate command-scoped timeouts and always-grade
settlement and have not been re-run.
