# Benchmarks for the recovery study

The recovery study branches an agent's run from an earlier step: it brings back
the files of the agent's container at that step, and the task's hidden tests
grade the branch when it ends. A state is files, never a running process or
another container, so a task is usable only if its hidden tests grade files
the agent left in its own container. This page records what was checked on the
measurement VM (Harbor 0.23.0, 32 vCPUs, 125 GiB, 124 GB disk) on 2026-10-07,
before any model was run.

| | SWE-Bench Pro V2 | Terminal-Bench 3.0 | SWE-bench Live |
| --- | --- | --- | --- |
| Source | github.com/scaleapi/SWE-bench_Pro-os, `v2/tasks` at 66f9276 (tag v2.0.0) | Harbor registry `terminal-bench/terminal-bench@3.0.0` | Hugging Face `SWE-bench-Live/SWE-bench-Live` |
| Harbor format | yes (task schema 1.4) | yes | no: needs a converter |
| Tasks | 642 | 74 | 1,888 (full), 1,000 (test), 500 (verified), 300 (lite) |
| Graded on files only | all 642 | 58 (rule below) | yes in principle (per-instance test commands) |
| Images | prebuilt, `ghcr.io/scaleapi/swe-bench_pro-v2:<instance>`, anonymous pulls | built locally from each task's Dockerfiles | prebuilt, `starryzhang/sweb.eval.x86_64.<instance>` |
| No-model check here | reference patch 4/4 solved, empty patch 4/4 unsolved | reference 3/4 solved, empty 4/4 unsolved | not run |

## SWE-Bench Pro V2

The public set as re-released on 2026-09-22: 642 tasks from 11 repositories
(89 of the original 731 were dropped as invalid), in Harbor's format. It is not
in Harbor's registry: `scale-ai/swe-bench-pro` and `cais/swebenchpro` there are
the original 731 tasks. Get it from the repository:

    sudo git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/scaleapi/SWE-bench_Pro-os /root/recovery-bench/swebench-pro-os
    sudo git -C /root/recovery-bench/swebench-pro-os sparse-checkout set v2

The tasks take 65 MB. The release also ships them as
`swe-bench-pro-v2.0.0-tasks.zip` with a `SHA256SUMS`, and its own gate (Harbor
0.22 on Modal): the reference patch solves 642 of 642, the empty patch 0.

Every task: agent limit 3,000 s, verifier limit 3,000 s, 1 CPU, 4 GiB memory,
10 GiB storage, no network during the agent phase, public network for the
verifier. The metadata gives every task the same difficulty (`medium`); the
only published difficulty signal is HARD-51, the 51 tasks failed by at least
two of five model families.

**Grading.** The verifier runs in the agent's container after the agent ends.
It restores every test file to the base commit, applies the hidden test patch,
runs the task's selected test files with the repository's own run script and
passes the task only if every fail-to-pass and pass-to-pass test passes.
Services a suite needs are started by that script (NodeBB's starts Redis).
Nothing depends on a process the agent left running, so all 642 tasks
qualify. The release's recommended protocol re-grades the agent's diff on a
fresh image (`v2/tooling/patch_replay.py`), which likewise reads files only.

**Images.** Compressed, an image is 0.54 to 6.52 GB (median 1.13). Tasks of a
repository share layers: all 642 hold 544 GB of distinct compressed layers,
0.32 GB per task for ansible up to 2.59 GB for protonmail/webclients. Unpacked,
the four images pulled here are 2.8 to 4.4 times their compressed size. This
host's Docker uses the containerd image store, which keeps the compressed
layers as well as the unpacked ones, so a task needs about 4 to 4.5 times its
share of compressed layers on disk.

## Terminal-Bench 3.0

    sudo /root/taste-harbor-20260927/venv/bin/harbor download \
        terminal-bench/terminal-bench@3.0.0 -o /root/recovery-bench

74 tasks (content `sha256:a32a6187...6da3`, 509 MB with their data). The
registry's latest is 4.0.0 (66 tasks), with a 63-task CPU-only subset
`terminal-bench/terminal-bench-cpu-only`; neither was examined here. No task
has a prebuilt image: Harbor builds each task's environment and, for its
verifier, a second image from `tests/Dockerfile`.

**Rule.** Every TB 3.0 task runs its verifier in a separate container
(`environment_mode = "separate"`). Harbor collects the task's declared
artifacts, stops the agent's container, and only then starts the verifier,
which sees those files and nothing else. A process the agent left running in
its container therefore cannot be graded. A task still fails the study's
requirement if

1. its environment has services besides the agent's container (sidecars):
   their state is neither in the agent's files nor restorable with them, and
   several of these tasks collect their artifacts from the running services
   (a database dump, a Redis snapshot, a Kafka log, results fetched over HTTP);
2. it needs a GPU, which the measurement VM does not have.

A collect hook that runs in the agent's own container and only reads files
(two tasks take a diff of the repository this way) is allowed. The test files
of the 58 remaining tasks were searched for network and process use (sockets,
HTTP clients, subprocesses, service managers): every server, emulator or
subprocess they contact is one the verifier starts itself, in its own
container. `scripts/recovery_candidates.py` applies the rule.

| Task | Category | Files only | Why not, or a note | CPUs / GiB | Agent limit (h) |
| --- | --- | --- | --- | --- | --- |
| atrx-vep-crispr | Science | yes |  | 2 / 4 | 5 |
| batched-eval-parity | ML | yes |  | 1 / 4 | 4 |
| biped-contact-dynamics | Science | yes |  | 2 / 8 | 4 |
| bun-sourcemap-leak | Software | yes |  | 1 / 2 | 0.5 |
| cad-model | Hardware | yes | its reference solution fails here (below) | 1 / 2 | 2 |
| cargo-flight-dispatch | Operations | yes |  | 2 / 4 | 1 |
| cli-2ph-simplex | Software | yes |  | 1 / 2 | 0.7 |
| coq-block-bound | Science | yes |  | 2 / 4 | 4 |
| ctr-optimization | Operations | no | sidecar services: api | 2 / 6 | 5 |
| cumulative-layout-shift | Software | no | sidecar services: barber-shop-data-backend | 4 / 8 | 3 |
| data-anonymization | Software | yes |  | 1 / 2 | 1 |
| distributed-dedup | Software | yes |  | 4 / 8 | 5 |
| embedding-drift-monitor | ML | yes |  | 1 / 2 | 2 |
| erp-procurement-planning | Operations | no | sidecar services: postgres, odoo | 4 / 8 | 1 |
| exam-pdf-eval | ML | no | needs a GPU | 4 / 14 | 4 |
| fin-saccr-rwa | Operations | yes |  | 2 / 4 | 2.5 |
| fix-uautomizer-soundness | Software | yes |  | 2 / 8 | 2 |
| foodstuff-beta-activity | Science | yes |  | 2 / 4 | 2.5 |
| formal-crypto | Security | yes |  | 1 / 2 | 2 |
| fp8-rmsnorm-gemm | ML | no | needs a GPU | 4 / 16 | 4 |
| freecad-impeller | Hardware | yes |  | 2 / 4 | 2.5 |
| freecad-platform-drawing | Hardware | yes |  | 2 / 4 | 2.5 |
| freecad-spring-clip | Hardware | yes |  | 2 / 4 | 2.5 |
| freight-dispatch-shift | Operations | no | sidecar services: event-feed | 2 / 4 | 2 |
| glycan-ms2-elucidation | Science | yes |  | 2 / 4 | 2.5 |
| gpt2-codegolf | ML | yes |  | 1 / 8 | 5 |
| gsea-proteomics | Science | yes |  | 2 / 4 | 2.5 |
| heat-pump-warranty | Operations | no | sidecar services: warranty-portal, asset-ledger, document-vault, returns-ledger, compliance-ledger, warranty-inbox | 2 / 4 | 2 |
| hof-topology-interpenetration | Science | yes |  | 2 / 8 | 2.5 |
| html-js-filter | Security | yes |  | 1 / 4 | 1 |
| ico-path-patch | Security | yes |  | 2 / 4 | 1.5 |
| interleaved-vigenere | Security | yes |  | 1 / 2 | 4 |
| intrastat-meldung | Operations | no | sidecar services: odoo, compliance-hub, idev, services, dms | 2 / 8 | 3 |
| jax-speedrun-gpu | ML | no | needs a GPU | 16 / 32 | 5 |
| ks-solver-cpp | Science | yes |  | 4 / 8 | 4 |
| kv-live-surgery | Software | no | sidecar services: loadgen | 2 / 4 | 1 |
| lake-temp-glm | Science | yes |  | 4 / 8 | 3 |
| layout-config-recreation | Media | yes |  | 2 / 4 | 1 |
| layout-config-recreation2 | Media | yes |  | 2 / 4 | 5 |
| lean-midpoint-proof | Science | yes | artifact: a diff of the repository, taken in the agent's container | 1 / 2 | 4 |
| legacy-utility-triage | Operations | no | sidecar services: legacy-workstation, legacy-app | 2 / 8 | 1.5 |
| live-database-cutover | Software | no | sidecar services: mysql-db, redis, postgres-db, customer | 16 / 16 | 2 |
| math-eval-grader | ML | no | needs a GPU | 8 / 16 | 4 |
| medical-claims-processing | Operations | no | sidecar services: playwright-mcp, workspace | 2 / 10 | 1.5 |
| memcached-backdoor | Security | yes |  | 4 / 12 | 2 |
| mp-checkpoint-consolidation | ML | yes |  | 2 / 4 | 2 |
| music-harmony | Media | yes |  | 2 / 4 | 2 |
| mvcc-lsm-compaction | Software | yes |  | 1 / 4 | 4 |
| nextjs-performance | Software | no | sidecar services: warehouse-api | 2 / 4 | 2 |
| ontology-kg-querying | Software | yes |  | 1 / 2 | 4 |
| payments-pipeline-fix | Software | no | sidecar services: seeder, kafka, customer | 4 / 8 | 2 |
| photonic-waveguide-routing | Software | yes |  | 1 / 2 | 3 |
| pretrain-shard-corruption | ML | yes |  | 2 / 8 | 2 |
| production-planning | Operations | yes |  | 2 / 4 | 1 |
| protein-autointerp-disulfide | Science | yes |  | 4 / 8 | 2 |
| react-lead-form | Software | yes |  | 1 / 2 | 2 |
| retro-console-soc | Hardware | yes |  | 2 / 4 | 4 |
| risk-scorer-replay | ML | yes |  | 2 / 2 | 2 |
| roy-polymorph-cn | Science | yes |  | 2 / 4 | 2.5 |
| rs-archive-clone | Software | yes |  | 4 / 4 | 4 |
| satb-audio-transcription | Media | yes |  | 4 / 8 | 2 |
| session-window-debug | Software | yes |  | 2 / 4 | 2 |
| sglang-qwen-burst | ML | yes |  | 2 / 6 | 2 |
| shadow-relay | Security | yes | artifact: a diff of the repository, taken in the agent's container | 2 / 4 | 2 |
| sound-change-cascade | Science | yes |  | 2 / 4 | 5 |
| takens-embedding-lean | Science | yes | 8-hour agent limit | 4 / 16 | 8 |
| telecom-entity-resolution | Software | yes |  | 1 / 4 | 2.5 |
| uefi-bootkit | Security | yes | the verifier boots the artifact disk image in QEMU | 2 / 8 | 2 |
| vba-userform-port | Software | yes |  | 2 / 8 | 2 |
| vf2-speedup-networkx | Software | yes |  | 1 / 4 | 2 |
| vllm-deepseek-streaming | ML | yes |  | 2 / 4 | 2 |
| vpp-loss-divergence | ML | yes | artifacts are library directories under site-packages | 2 / 8 | 2 |
| wal-recovery-ordering | Software | yes |  | 2 / 4 | 2 |
| wdm-design | Science | yes |  | 2 / 4 | 5 |

The 58 tasks ask for 115 CPUs and 282 GiB in all; their agent limits run from
0.5 to 8 hours (median 2.25). The published leaderboard puts the best agents
at about 34% and GPT-5.6 Luna under Codex at 14.3%, so few of these tasks are
expected to be of middle difficulty for a small model.

## SWE-bench Live

Not in Harbor's registry, not among Harbor's adapters, and not published in
Harbor's format by its maintainers. Using it here needs a converter: one task
directory per instance with the instance image as `docker_image`, the problem
statement as the instruction, and a verifier that restores the test files,
applies the test patch, runs the instance's `test_cmds` and parses the log
with the instance's parser, following the benchmark's own resolve rule
(`taste/benchmarks/swebenchlive.py` implements that rule). The converted tasks
would then need the same two-sided gate as SWE-Bench Pro V2 (reference patch
solves, empty patch does not) before use. SWE-Bench Pro V2 alone supplies more
candidates than the calibration needs, so this was not done.

## No-model checks on the measurement VM

Harbor's built-in agents, `oracle` (applies the reference solution) and `nop`
(does nothing), run with `harbor run -p <tasks> -a oracle|nop`. Neither calls a
model. Times are per trial, in seconds.

| Benchmark | Task | Reference | Empty | Environment start (cold / warm) | Verifier |
| --- | --- | --- | --- | --- | --- |
| SWE-Bench Pro V2 | ansible-34db57a4 | solved | unsolved | 261 / - | 88-89 |
| SWE-Bench Pro V2 | NodeBB-05f22361 | solved | unsolved | 261 / 16 | 12-14 |
| SWE-Bench Pro V2 | navidrome-d613b193 | solved | unsolved | 262 / - | 25-26 |
| SWE-Bench Pro V2 | qutebrowser-77c35579 | solved | unsolved | 262 / 16 | 12 solved, 298 unsolved |
| Terminal-Bench 3.0 | foodstuff-beta-activity | solved | unsolved | 277 / - | 103-108 |
| Terminal-Bench 3.0 | music-harmony | solved | unsolved | 268 / - | 110-143 |
| Terminal-Bench 3.0 | sound-change-cascade | solved | unsolved | 277 / 50 | 80-191 |
| Terminal-Bench 3.0 | cad-model | **not solved** | unsolved | 268 / 51 | 30-83 |

A cold start includes pulling the image (SWE-Bench Pro V2, four images at once,
while other jobs ran) or building it (Terminal-Bench); a warm start reuses it.
A Terminal-Bench verifier includes building and starting its own container.

- cad-model's reference solution installs the current `build123d` from PyPI,
  which no longer imports with the OCP library in the image
  (`No module named 'OCP.collections'`). The task fails its own reference
  check here and should be left out until fixed upstream.
- A verifier can take far longer on an unsolved attempt than on a solved one:
  qutebrowser's hidden tests took 298 seconds on the unchanged code against
  12 after the reference patch. A calibration's failed attempts can use much
  of the 3,000-second verifier limit.
- Five trials of the first runs (cad-model and sound-change-cascade in both
  Terminal-Bench jobs, qutebrowser's empty patch) ended with `CancelledError`:
  the host's unattended upgrade (below) restarted their units. They were run
  again; where a task has two times, they are the two runs.
- Every Terminal-Bench 3.0 candidate should pass the same two-sided check
  before it is calibrated: it costs no model calls, and builds the images the
  calibration needs anyway.

## Disk and concurrency

The VM's disk is the limit. It had 59 GB free; the 242 SWE-Bench Pro V2
candidates below hold 208 GB of distinct compressed layers, about 0.9 TB on
this Docker (here 14 GB of compressed layers took 45 GB unpacked besides).
A data disk of 1 TiB for Docker's and containerd's data holds the SWE-Bench
Pro V2 candidates' images; 2 TiB holds every candidate's images at once and
leaves room for the study's checkpoints. On the present disk the calibration
must run in waves of about ten tasks, removing each wave's images after it. Pulls are slow enough to plan for: the
four smoke images (3 GB compressed) took 4.4 minutes together while other jobs
ran, which for the 208 GB would be about five hours. Pull ahead of the run,
several images at a time.

An agent-alone trial asks the host for little beyond its task container
(mini-swe-agent and Taste's processes wait on model calls). A SWE-Bench Pro V2
container is limited to 1 CPU and 4 GiB, so 24 trials at once use at most 24
CPUs and about 100 GiB, leaving room for image pulls, unpacking and Harbor.
Terminal-Bench 3.0 tasks ask for 2 CPUs and 4.9 GiB on average: 12 at once.

Ubuntu's unattended upgrades and needrestart restart every service after a
library upgrade, the running trials' units included: on 2026-10-07 at 04:35
UTC an OpenSSL upgrade did so and ended the trials then running. Hold them
during measurement (`systemctl disable --now apt-daily-upgrade.timer`, or
`$nrconf{restart} = 'l'` in `/etc/needrestart/conf.d/`).

## Calibration

`data/recovery/calibration-candidates.json` holds 300 candidates, and pins
each dataset they come from by a digest of its files (as the Terminal-Bench
2.1 split does):

- 242 SWE-Bench Pro V2 tasks, left out HARD-51, stratified by language
  (Python, Go, JavaScript/TypeScript, by the files the reference patch changes)
  and by the size of the reference patch (terciles within the language), seats
  in proportion to each stratum, taken in the order of SHA-256 of the salt
  `taste-recovery-calibration-2026-10-07` and the task name;
- the 58 Terminal-Bench 3.0 tasks above that grade files only (cad-model to be
  dropped if its reference check still fails).

The record is made by

    python3 scripts/recovery_candidates.py --pro /root/recovery-bench/swebench-pro-os/v2 \
        --tb3 /root/recovery-bench/terminal-bench --pro-seats 242 \
        --salt taste-recovery-calibration-2026-10-07 --out data/recovery/calibration-candidates.json

As root, each part's tasks are linked into a directory of their own (hard
links, on the same disk):

    cd /root/recovery-bench && mkdir -p runs/cal-pro runs/cal-tb3
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("\n".join(r["swebench_pro_v2"]["tasks"]))' \
        <checkout>/data/recovery/calibration-candidates.json | while read -r t; do cp -al swebench-pro-os/v2/tasks/$t runs/cal-pro/; done
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("\n".join(r["terminal_bench_3"]["tasks"]))' \
        <checkout>/data/recovery/calibration-candidates.json | while read -r t; do cp -al terminal-bench/$t runs/cal-tb3/; done

mini-swe-agent alone, GPT-6 Luna, two runs per task, as Taste's agent-alone
arm (`run-harbor.sh` runs the job in its own systemd unit; `JOBS` keeps it out
of other projects' job directories):

    sudo MODEL=azure/gpt-6-luna JOBS=/root/recovery-bench/jobs infra/azure/run-harbor.sh \
        cal-pro /root/recovery-bench/runs/cal-pro -k 2 -n 24 --max-retries 2 \
        --ak agent=mini-swe-agent --ak services=none --ak spend_cap_usd=1 \
        --ak worker_max_calls=1000 --ak max_commands=2000 --ak reply_reserve_seconds=10

    sudo MODEL=azure/gpt-6-luna JOBS=/root/recovery-bench/jobs infra/azure/run-harbor.sh \
        cal-tb3 /root/recovery-bench/runs/cal-tb3 -k 2 -n 12 --max-retries 2 \
        --ak agent=mini-swe-agent --ak services=none --ak spend_cap_usd=1 \
        --ak worker_max_calls=1000 --ak max_commands=2000 --ak reply_reserve_seconds=10

An agent alone may spend the trial's whole cap (`spend_cap_usd`); $1 stops a
run that loops long before it matters to the estimate below. Harbor retries a
trial only for infrastructure errors: a timed-out agent or verifier is graded
as it stands. The two jobs together would ask for 36 trials at once: run them
one after the other, or halve both. With GPT-5.6 Luna instead
(`MODEL=azure/gpt-5.6-luna`), the same commands cost about twice as much.

Then, as root on the VM:

    python3 scripts/calibration_report.py /root/recovery-bench/jobs/cal-pro /root/recovery-bench/jobs/cal-tb3 \
        --json cal.json --keep kept.txt

gives, per task, attempts and successes, the agent's model calls in each run,
dollars and wall-clock, and keeps a task when it solved 20-80% of its graded
attempts and its runs had a median of at least 15 steps. With two runs, only
1 of 2 is in that range: a task whose true rate is 0.5 shows it half the time,
one at 0.2 or 0.8 a third of the time. If fewer than ~120 tasks are kept, a
third run on the tasks at 0 of 2 or 2 of 2 with long runs moves some into
range (1 of 3, 2 of 3); a second wave of SWE-Bench Pro V2 tasks can favour the
strata that kept the most.

**Expected cost and time.** On Terminal-Bench 2.1's tuning tasks (133 runs),
mini-swe-agent with GPT-5.6 Luna averaged 5,700 input tokens (83% cached) and
290 output tokens per step, nearly every uncached token was written to the
cache, and a run's input grew with the square of its steps (median 9 steps,
$0.0076 a run). GPT-6 Luna has not been measured here; taking the same
behaviour at its prices ($0.10 input, $0.01 cached, $0.125 cache write and
$0.50 output per million tokens) and SWE-Bench Pro's larger observations, a
run costs about $0.02-0.03 at 30 steps, $0.05-0.07 at 60 and $0.11-0.14 at
100. The 484 SWE-Bench Pro V2 trials should come to $15-35 and the 116
Terminal-Bench trials to $5-20: about $30 in all, and never more than $600
under the cap. At 24 trials at once and about 10 minutes a trial, the
SWE-Bench Pro V2 part takes 3.5-4 hours once its images are on disk; the
Terminal-Bench part 3-4 hours if most runs end within half an hour, and up to
a day if many use their multi-hour limits.
