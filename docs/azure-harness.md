# Azure OpenAI harness: reference notes

Reference for the worker harness that runs on Azure OpenAI's Responses API, and
for the pieces that let it work inside a benchmark's task container: the
terminal broker, the trial owner and the Harbor agent. It is dense on purpose
and assumes [architecture.md](architecture.md). For running a benchmark, start
with [`infra/azure/benchmarks.md`](../infra/azure/benchmarks.md).

## The worker harness

The Claude Agent SDK harness remains available through
`taste.brains.worker_entrypoint` and `taste.brains.worker_launch`. The opt-in
Azure Responses harness uses `taste.brains.azure_worker_entrypoint` and the
`worker_command_factory` in `taste.brains.azure_worker_launch`, passed to
`SubprocessLauncher`. It does not import the Claude Agent SDK. Install the
`openai` extra and supply `AZURE_OPENAI_BASE_URL` and `AZURE_OPENAI_API_KEY` in the
launch environment; each immutable Assignment must carry its matching
`AzureWorkerPolicy`, worker/monitor budgets, call limits and absolute deadline.
Personal OpenAI and Anthropic credentials are not fallback routes.

The Azure worker supports confined artifact tools, explicit feedback acceptance,
private model-call accounting, exact-state monitor certification, and coordinator
delivery. `compose_azure_central_runtime` in `taste.brains.azure_central_host`
wires the Azure planner and worker launcher together. Its `AzureExecutionPolicy`
pins the routes, call limits, worker/monitor budgets, pricing table and absolute
deadline into the immutable goal. The planner must propose assignments matching
that policy. The host owns and closes its private planner SDK client.

For a separate goal process, `prepare_azure_goal_process` in
`taste.brains.azure_goal_entrypoint` returns the exact `GoalProcessInput` without
model calls. Persist its bytes outside task write access, hash them with SHA256,
and use `azure_goal_command(input_path, digest, mode="run")` to build the isolated
command. `mode="settle"` reconciles the same admitted goal without provider
credentials and refuses new model calls or worker launches. The caller must own an outer process scope that enforces a
hard deadline and confirms all descendants have stopped. The historical Claude
goal and worker entrypoints remain separately available.

`OwnedProcessScope` can deliver private files through systemd `LoadCredential`.
Add `ScopeCredential` name/digest descriptors to its `ScopeSpec` and pass the
matching byte mapping to `OwnedProcessScope.create(..., credentials=...)`.
The controller snapshots those bytes outside worker write access and validates
the files and every path ancestor before launch. Only names and digests enter
the saved specification; secrets never enter service arguments or environment
values. The service reads its read-only copies through `CREDENTIALS_DIRECTORY`.
The owner retains its private snapshots for recovery and must remove them when
the trial is settled. Credential-free scope records retain their existing format
and digests. `scripts/check_scope_credentials.py` checks actual unprivileged
delivery, deadline cleanup, lost-acknowledgement recovery and the Azure credential
loader using dummy bytes. The loader admits either a private service-owned copy
or systemd's root-owned copy with exactly one ACL reader for the service UID.

For the standalone Azure command, encode the exact prepared input's API key and
optional terminal issuer with `encode_azure_goal_credentials` in
`taste.brains.azure_goal_credentials`. Deliver those bytes as the scope credential
`taste-azure-goal.json` and call `azure_goal_command(..., systemd_credentials=True)`.
The goal validates the private input binding and probes terminal issuer readiness
before the first paid plan. A failed or cancelled probe cannot dispatch a goal.
Workers receive their own assignment grants and Azure authentication; the
coordinator's credential-directory hint is removed at their process boundary.
Preparation and settlement require no provider or terminal credentials.

`AzureGoalService` in `taste.brains.azure_goal_service` binds these operations to
an `OwnedProcessScope` and a durable result exchange. `create(...)` admits a
fresh `prepare`, `run`, or `settle` service; `run(timeout_seconds=...)` launches
it once, while `recover()` only drains and inspects the original service.
Results are read after confirmed whole-scope drainage, with exact input,
operation, goal identity and cost-accounting checks. A missing result remains
incomplete; after drainage, use a separate credential-free settlement operation.
An ambiguous launch stays fenced even if a result file exists. The owner must
protect input, scope and exchange paths from model writes. Only a `run` service
receives the private Azure credential. Place the fresh workspace in a separate
service-owned state directory: memory worktrees are created beside the workspace.
Preparation rejects an unwritable parent before initializing Git.
`grading_ready()` admits settled bounded
outcomes with known spending; benchmark reward still comes from the verifier.
`scripts/check_azure_goal_service.py` exercises real Linux services for completion,
a lost provider reply and SIGKILL during dispatch using mocked HTTP and no paid
calls. Its independent `--cleanup-only` command drains the recorded scopes.

An existing or partly initialized worker run is refused by the fresh-launch
entrypoint; it never resets the allowance on restart. A native Harbor agent now
runs benchmark tasks as goals with real models (next section). No full or scored
benchmark run has been made.

## Running a Harbor benchmark task

`taste.benchmarks.harbor_agent:TasteAgent` runs one benchmark task as a goal in
Harbor's standard Docker environment, as the task is published: no environment
subclass, extra Compose file, mount or override. The coordinator, its workers
and their monitors run on the host; only shell commands enter the task
container. Procedure, settings and what has been validated are in
[`infra/azure/benchmarks.md`](../infra/azure/benchmarks.md).

- Every worker is shown the task verbatim before its assignment. The
  coordinator writes the final reply. When time, budget, the generation bound
  or planner failures end a run first, it writes a closing reply from the
  evidence it has, inside a reserved part of the trial's time.
- A command that outlives its timeout is ended alone, with its child processes.
  The container stays usable.
- The trial leaves one ATIF trajectory in the order things happened: the
  instruction, every worker step with one tool call each, then the reply.
  Planner and monitor calls stay in the private record with every model
  receipt and the terminal ledger.
- The trial is handed to the benchmark's verifier whatever this system's own
  audit found. Missing evidence or unsettled cost is recorded as `audit_flags`
  in the trajectory and the trial metadata. It is never used to withhold a task
  from grading. A run cancelled by the benchmark's own time limit is sealed
  first and graded as a timeout.

For [Errata Bench](https://errata-bench.vercel.app/docs) this has been run with
real models on a handful of tasks (see the guide for the list and the
measurements). Errata's own unpaid admission check counts those trials as
gradable and official. No full run and no paid grading has been made, and
Terminal-Bench is not supported: its official runs use a sandbox this terminal
transport cannot reach.

`AzureTerminalTrial` in `taste.benchmarks.azure_terminal_trial` owns an Azure goal
and an already admitted Docker task container. `create(...)` persists their
original identities and limits in a private controller directory. `run(api_key=...)`
prepares the goal, issues private terminal grants, runs it in a separate
unprivileged service, drains that service and settles without credentials before
sealing the live container for grading. Unknown spending and incomplete
evidence are recorded as `audit_flags` and do not withhold grading. The
container deadline may extend beyond the agent deadline to allow the verifier
and output downloads to finish; neither deadline is renewed on recovery.
Call `release()` to hand the sealed container to a runner that verifies and
removes it itself, or `close()` to stop it after grading or on failure. For
native Compose bindings,
cleanup also handles Harbor having removed the original container: it requires
an explicit daemon absence response and an empty original cgroup. A failed
daemon request or a live cgroup cannot establish successful cleanup.
An independent watchdog must call `cleanup_trial(directory)` after controller
death. It drains the saved scopes and original container, removes the private
launch credential and preserves pending terminal receipts without replay.
Drainage alone does not settle an interrupted goal's spending.

The controller-side `DockerTerminalBackend` in `taste.brains.docker_terminal`
implements bounded non-TTY terminal transport over an explicit local Docker
socket (API v1.51, Linux cgroup v2). It binds the full container ID, original
start time and either a reserved ownership label or Harbor's native Compose
project/main-service identity and exact image ID. A fixed admitted execution
user preserves Harbor's agent-user setting without a task override. It captures
bounded binary stream prefixes with
durable dropped-byte counts, and confirms exit status through Docker's exec API.
Stopping requires an empty original cgroup and a stopped container (or confirmed
removal for a native Compose binding).
Persist the original `DockerTerminalBinding` outside task write access for recovery.
The lifecycle owner must create the isolated container with restart disabled
and independently stop/remove it if the controller dies. Workers must not receive
the Docker socket. `scripts/check_docker_terminal.py` exercises this transport in
five serial disposable containers; its invocation documents the outside cleanup
requirement.

`scripts/check_native_harbor_terminal.py` checks native Harbor container labels,
configured non-root user, working directory, resource limits, standard log
mounts, shared verification and controller-death cleanup without an environment
subclass or extra Compose file. It is a synthetic terminal integration check;
it does not exercise model-only egress or claim an official benchmark score.

`TerminalService` and `TerminalClient` in `taste.brains.terminal_service` connect
workers to that broker over a bounded Linux Unix-socket protocol. Peer UID checks
and a private random bearer token bind each worker to its immutable assignment,
task container and original deadline. A command that outlives its timeout, or
whose caller disconnects, is ended alone with its child processes and recorded
as a completed receipt; a transport that cannot end one command stops the whole
environment instead. A completed request can be looked up without running it
again. The broker serializes all workers' terminal effects in the same container.

To admit terminal tools, set `AzureExecutionPolicy.terminal` to the public
`TerminalWorkerPolicy` and give `compose_azure_central_runtime` a trusted
`terminal_credential_provider`. For each prepared launch, that callback must
register the matching `TerminalCredential` with the service before returning it.
Credentials are installed in private files outside model-visible memory; the
worker verifies them and probes the service before paid admission. It receives
`terminal_exec` and `read_terminal_output`, with bounded pages and explicit
truncation counts. Files, packages and services in the container persist across
memory rollback. Omitting the terminal policy preserves artifact-only execution.

For a separate coordinator process, `TerminalIssuerClient` supplies this callback
over the same socket using a distinct private `TerminalIssuerCredential`.
The broker checks the coordinator UID, exact goal-input digest and terminal
policy, derives each actor from its assignment, and refuses changed checkpoints,
expired deadlines or new grants after sealing. Worker tokens cannot issue grants;
the issuer token cannot execute commands. A repeated issuance request returns
the same token only within that service lifetime. Service loss still requires
outside cleanup and recovery rather than restarting workers speculatively.

The standalone Azure goal CLI can load the issuer callback, but the outside
lifecycle owner must still create the task container and broker, enforce the process
deadline, drain agent processes, and call `TerminalService.seal_for_grading()`
before grading the live task. Sealing refuses an active or uncertain command,
permanently blocks new effects (including after broker recovery), and leaves
completed receipts readable. Task background services remain part of the live
environment. Call `close()` after grading or on failure; an independent owner
must still stop/remove the container if the controller dies.
`scripts/check_terminal_service.py` validates the broker with two serial
containers and unprivileged worker processes, including grading a fixture after
sealing and killing a worker during a command. These checks do not yet constitute
a Harbor adapter or an official Terminal Bench result.

The fixture scripts described from here on (`check_terminal_service.py`,
`check_terminal_broker.py`, `check_harbor_trial.py`, `check_azure_harbor.py`,
`check_harbor_separate.py`) were written before command-scoped timeouts,
plain-text terminal results and always-grade settlement. They have not been
re-run since, and at least three of them assert the earlier behaviour.
`check_docker_terminal.py` is current and was re-run against a real daemon.

`scripts/check_harbor_trial.py` exercises the pinned Harbor revision
`d611f10b15ffab8afb7b665b3e69e96773044fc8` in a separate server environment.
It runs a controlled local task through Harbor's actual Trial and verifier with
an unprivileged scripted worker. The success case requires reward `1.0`; the
worker-death case requires an exception and no reward. Every case requires broker stop
proof before container removal, an empty original cgroup, and retention of the
cached image. Run each mode in a bounded systemd unit with the independent
owner-label cleanup described in the script. Telemetry is disabled and no image
pulls or paid calls occur. This validates the grading/cleanup handoff for that
fixture.

`scripts/check_azure_harbor.py` connects the actual Azure planner, workers,
monitors, issuer and `AzureTerminalTrial` to the same pinned Harbor Trial and
official verifier. Only provider HTTP is mocked. Its modes cover completion,
worker death after a terminal effect, a lost planner reply, and verifier output
collection after the agent deadline. `killed-owner` kills the Harbor/controller
process during an active terminal command; run its independent `--cleanup-only`
command through systemd `ExecStopPost`, then use `--observe-owner-death` to check
drainage and retained uncertainty. Use a fresh 32-character hex owner token per
case and keep the Harbor and Azure Python environments separate. These controlled
fixtures do not yet admit arbitrary Terminal Bench tasks or produce benchmark
scores.

`scripts/check_harbor_separate.py` extends this lifecycle to a second, separately
owned verifier container. The actual Harbor collector transfers a regular file,
a directory containing binary data, and the conventional artifacts directory.
`ArtifactHandoff` requires successful bounded captures **and** a complete matching
Harbor collection manifest, then checks the captured bytes before verifier
uploads. Missing files, links, altered bytes and failures after a successful copy
prevent verifier creation. A valid reward is withheld if verifier cleanup fails;
zero is a valid reward, while missing and non-finite rewards remain errors.
The `kill-copy` and `kill-verifier` modes require independent `--cleanup-only`
execution through `ExecStopPost`, followed by `--observe-owner-death`. Both original
container identities must drain, even when the controller dies during artifact
capture or verification. The trusted test script is installed during controlled
fixture setup; this does not validate released task images or general task
admission. Each case uses the cached image, no task network and no paid calls.

The same fixture's `service-*` modes add a helper service with its own filesystem
and original Docker identity. The real terminal action writes to that service
over a private loopback connection; the main container has no external network.
Before helper artifacts are copied, `ArtifactHandoff.drain_main()` confirms the
original main container has stopped. Failed stop attempts, restarted services or
incorrect service names cannot yield valid evidence, even when Harbor continues
after an error. Receipts bind each capture to its service and original container;
limits apply across services, and overlapping verifier paths are rejected.
Helper cleanup must also complete before verifier creation. The modes
`kill-service-copy` and `kill-service-verifier` exercise independent cleanup after
controller death. Cleanup of a restarted helper retains the identity conflict
and records removal separately; the restarted service is never re-admitted for
collection. Released-task collection hooks, filtered/mode-preserving output
and general service topology still require admission and validation.

These fixtures export logs, artifacts and rewards through
`taste.benchmarks.output_snapshot`, with no host output mounts. It checks the
original Docker identity before and after a bounded archive download, validates
all entries before writing, and creates only ordinary files. Directory snapshots
require an empty private destination; individual files require an unused name in
a private parent. Links, special/sparse files, path escapes, duplicates, stale output
and excess data are refused. Defaults cap archives at 16 MiB, individual files
at 4 MiB, total file data at 8 MiB and entries at 1,024. These are explicit
admission limits, not silent truncation. Additional real fixture modes
`reward-symlink`, `reward-fifo` and `reward-oversize` require a verifier download
error and no reward. The controller must retain ownership of in-flight copies
through cancellation and must not grade a failed snapshot. Artifact handoffs also
cap aggregate file bytes and entries across all declared outputs. Source modes
and ownership are not applied to host files. Larger outputs, executable-mode
preservation, links, arbitrary destinations and filtered downloads need explicit
admission before running tasks that require them.
