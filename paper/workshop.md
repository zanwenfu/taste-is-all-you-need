# Twenty-Six Ways to Measure Nothing: A Failure Catalogue from Instrumenting an Agent Harness

*AgenticOS @ NeurIPS 2026 — regular paper draft v2. ANONYMISED — no author, repo, or host names before submission.*

---

## Abstract

An OS layer for agentic AI will standardise checkpointing, recovery, and
observability, and those designs can only be compared if they can be
measured. We set out to measure one contrast — whether rolling back to a
verified checkpoint leaves less previously-working behaviour broken than
repairing in place — and built the instrument it requires: an observational
timeline committed to git, exhaustive replay of held-out tests at every
observation, and detection attributed by coverage rather than co-occurrence.

What we report is the instrument's failure record. **Twenty-six defects, of
which twenty-one produced a plausible number rather than an error, and all
but one were present while a green test suite watched.** Six were invisible
on any developer machine *by construction*, because they depend on the
absence of ambient configuration a developer machine has. The defects
collapse into four mechanisms — ambient-environment dependence,
infrastructure failure conflated with measurement, final-state measurement of
event-shaped quantities, and output consumed without asserting its producer —
and we show each mechanism operating in public infrastructure other than
ours, including the official SWE-bench grader.

The capstone is the defect our own validation regime missed. After
nineteen fixes, every gate we had built passed — the negative control, the
positive control with recovered regressions, the liveness check, an exact
re-scoring reproduction — and the headline number was still wrong, because
the agent under test executed on the host in an uninstalled source tree while
every check we had validated the *measurement* path. A validation gate
certifies only the paths it exercises, and the path we missed was the one
the agent ran on.

---

## 1. Introduction

The agentic-systems community is converging on an OS layer: shared
abstractions for memory, scheduling, checkpointing, recovery. Choosing
between abstractions requires evidence, and this paper is about how much
harder producing that evidence is than it looks — not in the statistics, but
in the plumbing that produces the numbers the statistics consume.

Our motivating question is narrow. When a step in a long-horizon agent run
fails verification, deployed harnesses do one of two things: **repair in
place** (hand the model its own broken tree) or **roll back and retry**
(reset to the last verified checkpoint). The second discards work and pays
full uncached token rates on restart; its defence is that it leaves less
collateral damage behind. That claim is testable — if you can say, for every
run, *when* previously-working behaviour stopped working.

**Contribution.** We report an engineering result. Building an instrument
that can answer "when did this break?" for an agent run is a problem whose
failures are systematically silent: a broken instrument and a clean run
produce the same number. We contribute (i) the instrument — observational
checkpointing on a hidden git ref, exhaustive replay, coverage-based
attribution, and a validation gate with liveness controls; (ii) a
twenty-six-defect catalogue with the mechanism for each, collapsed into a
four-class taxonomy whose classes we show operating in public harnesses, not
only ours; and (iii) two design rules and one working practice that caught
most of them, cheap enough to adopt wholesale. We claim no result about
recovery policy: the experiment this instrument was built for had not run
validly at submission time, and §5 explains precisely why we know that.

---

## 2. What has to be measured

A **silent regression** is a test that passed at some observation of a run
and fails at a later one, with nothing in the harness reporting it. Three
properties make it hard to measure honestly.

**It is an event, not a state.** A regression introduced and then repaired
leaves no trace at the end of the run. An instrument that inspects the final
artifact — which is how regressions on this benchmark are measured in
published work [3] — records zero. We observed the concrete case: eight
assertions broke at observation 3 and passed again at observation 4, because
the harness's own rollback erased them. Final-state measurement reports that
nothing happened; the event was real.

**Its absence and the instrument's death are the same observation.** Zero is
what a clean run produces. Zero is also what a probe that cannot execute
produces, what a parser that matched nothing produces, and what an empty
timeline produces. This is the paper's central practical claim, and §4 is
twenty-six instances of it.

**Detection must be attributed, not co-located.** "The harness failed
something while the regression was open" credits whichever policy fails most
often with the best detection — the opposite of the quantity under study.
Attribution needs a causal join: a failing harness check and the broken
held-out test must exercise a file the agent changed.

---

## 3. The instrument

**Observational checkpointing.** Every mutating tool call writes a commit on
a git ref the agent cannot enumerate, built from a private index, so the
agent's own `git status` and `git diff` are byte-identical with the
instrument on or off. Every policy gets the same observation granularity —
the only way detection latency is comparable across policies.

**Exhaustive replay, not bisection.** Held-out tests are replayed at *every*
observation, inside the instance's pinned container image. An earlier
version bisected for the onset, which assumes monotone verdicts. That
assumption is violated *differentially by policy* — non-monotonicity is the
treatment — so bisection would have recorded only the regressions the
recovering policy failed to fix, silently flattering it.

**Attribution by coverage.** A harness failure detects a regression only when
some failing harness check and the broken held-out test both exercise a file
the agent modified at that observation. Both the attributed and the loose
co-occurrence figures are always reported.

**Gate 0.** Five pre-set checks: negative control (clean runs must measure
zero), positive control (injected regressions — including *recovered* ones —
must be found), flake screen, unknown-rate ceiling, and **baseline
liveness**: a clean run's probes must actually answer PASS. The last exists
because of §2's second property; before it existed, an instrument whose every
probe errored scored a perfect negative control.

**The re-scoring control.** Because observations are commits, an archived run
can be re-measured under a changed instrument with no model calls: same
trajectory, only the instrument varies. When two pilots disagreed sharply (14
episodes, then 0) after a change that touched the measurement path,
re-scoring the first pilot's archived timelines under the second's instrument
reproduced its 8 and 6 episodes *exactly* — proving detection had not
regressed, and localising the change to the runs. §5 returns to what this
control can and cannot certify.

---

## 4. Twenty-six ways to measure nothing

Every row is a real defect from this work. **S** = silent: it produced a
plausible number rather than an error. **G** = present while the project's
test suite was green (row 16 *is* the suite, marked —). **H** = findable
only on a clean host, by construction.

### 4.1 The catalogue, by mechanism

**Class A — ambient-environment dependence.** The code asks the machine a
question and gets a different answer on a different machine.

| # | Defect | Consequence | S | G | H |
|---|---|---|---|---|---|
| A1 | Shadow commits inherit the machine's git identity | timeline silently empty on any clean machine | ✓ | ✓ | ✓ |
| A2 | Gate 0's probe invoked bare `python` | every probe exit 127 on clean Ubuntu | ✗ | ✓ | ✓ |
| A3 | Unpinned SDK resolved a different major version | two machines ran different code; every model call failed | ✗ | ✓ | ✓ |
| A4 | Test fixtures verified with bare `pytest` from PATH | 15 tests fail on a clean host, presenting as harness defects | ✗ | — | ✓ |
| A5 | A benchmark scorer invoked bare `python` | score 0.0 manufactured by a missing interpreter | ✓ | ✓ | ✓ |
| A6 | **Agent executed on the host; measurement in the pinned image** | the capstone — see §5 | ✓ | ✓ | ✓ |

**Class B — infrastructure failure conflated with measurement.** The
instrument's zero and the defect's zero are the same number.

| # | Defect | Consequence | S | G | H |
|---|---|---|---|---|---|
| B1 | Output marker used shell `:` (prints nothing) | a perfect 13/13 run scored as 13 errors | ✓ | ✓ | |
| B2 | Results on stderr, markers on stdout | one framework's results fell outside the parsed slice | ✓ | ✓ | |
| B3 | Probe diffed against a deliberately-deleted commit | every graded test a hole → "0 regressions" | ✓ | ✓ | |
| B4 | Interrupted mirror clone accepted with zero commits | later instances die far from the cause | ✓ | ✓ | |
| B5 | Container workdir unvalidated | every exec exit 127, OCI error on stdout | ✓ | ✓ | |
| B6 | Isolation removed loopback, not just the internet | 23.5% of the oracle dead; one instance 812 tests | ✓ | ✓ | |
| B7 | Provider cache served force-removed containers | every later cell of an instance replays against a dead container and **scores clean** | ✓ | ✓ | |
| B8 | Per-observation tree reset reverted image build-time source edits | a whole repo family's oracle dead, disguised as flake | ✓ | ✓ | |

**Class C — final-state measurement of an event-shaped quantity.** The
number measured is not the quantity defined.

| # | Defect | Consequence | S | G | H |
|---|---|---|---|---|---|
| C1 | Bisection assumed monotone verdicts | recovered regressions invisible; recovering policy flattered | ✓ | ✓ | |
| C2 | The declared event unit was implemented nowhere | pipeline counted per parametrised test; λ̂ off by up to 2.7× | ✓ | ✓ | |
| C3 | Rollbacks produced no observation | the recovering arm's recoveries invisible to the timeline scoring it | ✓ | ✓ | |
| C4 | "Persists past the step boundary" defined nowhere | one legal reading excludes the events the design exists to count | ✓ | ✓ | |

**Class D — output consumed without asserting its producer.** Every
consumer trusted; no producer verified.

| # | Defect | Consequence | S | G | H |
|---|---|---|---|---|---|
| D1 | Audit-trail field never populated | every checkpoint reported 0 tool errors, always | ✓ | ✓ | |
| D2 | Malformed planner output crashed the run | 33% of runs lost, scored as task failures | ✗ | ✓ | |
| D3 | Dependency install counted as agent work | 3,640 of 3,641 changed files; attribution vacuous | ✓ | ✓ | |
| D4 | Join preferred the first observation over the last | failures pinned to the earliest tree under the fine grid | ✓ | ✓ | |
| D5 | Executed config rebuilt from a name | manifest describes a run that never happened | ✓ | ✓ | |
| D6 | Git handles never released | sweeps die of fd exhaustion after ~400 cells — loss concentrated on the back half, dropped from the denominator | ✓ | ✓ | |
| D7 | Console re-walked the full tree per run | answers first request, times out on all others: presents as a dead server | ✗ | ✓ | |
| D8 | Detection events never carried failing-test ids while attribution joined on them | "silent vs detected" — the title construct — structurally UNKNOWN on every real run | ✓ | ✓ | |

Totals, computed from the tables: **21 of 26 silent; 25 of 26 present under
a green suite (the 26th being the suite itself); 6 of 26 findable only on a
clean host.**

### 4.2 The two design rules that caught most of them

**Infrastructure failure is a missing observation, never a test failure.** A
container that will not start, a patch that will not apply, a timeout, an
unparseable log — all yield a typed `error`, never `fail`. This single rule
converted B1, B2, B3, B5 and B6 from *fabricated regressions* into *visible
holes*. Without it, each would have reported contamination at every
observation of the affected instances.

**Zero events must be distinguishable from a dead instrument.** Gate 0's
liveness check asserts a clean run's probes actually answer PASS. On first
contact with a clean cloud host, this check failed loudly — negative control
`dead` on every sample — on defect A2, which would otherwise have been
certified as a clean result on the machine about to run the real experiment.

### 4.3 The working practice that found the rest

**Never consume a measurement's output without first asserting the
measurement succeeded.** D6 hid for weeks because a fixture consumed a
sweep's output field without checking the sweep's status; the real error —
file-descriptor exhaustion — sat unread in the record while the failure
presented as an unrelated path error. Two assertions turned a mystery into
its own error message. Nearly every row above is some component violating
this rule; it is also how the official SWE-bench grader could be induced to
mark unresolved patches resolved by forged stdout [5].

---

## 5. The capstone: every gate passed, and the number was still wrong

We ran three exploratory pilots on SWE-bench Verified (80 runs, ~$110,
development slice only, permanently excluded from any confirmatory frame).
The measured event rate was far below the pre-declared gate for proceeding:
λ̂ = 0.11 events/run against a gate of 0.30, with 2.5% of runs bearing any
event against a gate of 25%. Twenty-eight of forty runs in the final pilot
completed zero plan steps. The gates and event unit had been fixed in
advance; the re-scoring control confirmed detection had not regressed. We
drafted the conclusion the pre-declared decision rule pointed to: the
benchmark's oracle offers too little opportunity per run; switch substrates.

The conclusion was wrong, and the way it was wrong is the paper.

The harness materialised each task as a bare source checkout on the host and
ran the agent there — while every probe, every graded verdict, every
validation gate ran inside the task's pinned container image, where the
project is built and installed. On the host, `import matplotlib` in that
checkout *succeeds*: the working directory shadows the installed package,
and an uncompiled source tree imports as a namespace package with a garbage
version. Anything touching a compiled extension then fails — and the
failure reads as *the agent's work being wrong*. Of 31 runs whose failure
named a command, 30 required the environment that existed only inside the
image; 26 of the 28 zero-step runs trace to this. The agent was verified
against a library it never built, in an environment the benchmark never
grades.

We therefore withdraw, explicitly: (1) "the zero is a property of those
runs" as a claim about the benchmark — it is a claim about our execution
path; (2) "the substrate is the binding constraint"; (3) the triggered
substrate switch. The pre-declared rule behaved correctly on the data it was
given; the *inference* attached to it did not survive the data's provenance.
No number in this paper is evidence about recovery policy, and no resolve
rate is reported because none was validly measured.

What survives is the control that let us say all this precisely. The
re-scoring reproduction proved the instrument's detection was stable across
its own changes — it correctly localised the anomaly to the runs. What it
could not certify is that the runs were sound, because no gate exercised the
agent's execution path: Gate 0 validated that the *probes* were alive in the
*image*, and nothing validated that the *agent* was alive in its
environment. **A validation gate certifies only the paths it exercises.**
The seam that fixes this is architectural, and it is the same seam the
instrument already uses: the agent must execute where it is measured.

### 5.1 The seam, built and checked the way the catalogue teaches

We built that seam — agent tools and Monitor checks execute inside the
pinned image, with bidirectional file coherence between the host tree the
instrument observes and the container tree the commands run in — and
validated it with the test the catalogue's hindsight demands: a *golden
check* that drives the benchmark's own gold patch through the real tool
path at $0 of model spend, and requires the official grader to return
``resolved=True``; a null run must return ``False``. On a real instance the
routed pipeline grades the gold patch resolved, all graded tests passing.

The check caught three more defects before they could cost anything —
a router aimed at a default path its sandbox did not use (which silently
created a real ``/testbed`` on the development host), a stale ledger that
skipped both halves of the check while the summary read as if they ran, and
a grading container that inherited the measurement's network isolation:
four of the instance's graded timeout tests need a network stack to time
out on, the official harness grades with the network up, and every resolve
rate would have read a few points low, forever, under conditions nobody had
chosen on purpose. Each was found by an assertion written before the run,
not by a person reading output — which is the paper's method reduced to
practice.

---

## 6. What this means for an agent OS

If observability is to be an OS-layer primitive, our experience compresses
into five rules. None is expensive; all were learned at retail price.

> **What to steal**
> 1. **Type your silence.** "Could not measure" must be unrepresentable as
>    "measured nothing" — in the schema, not in convention.
> 2. **Prove the instrument alive, not just non-lying.** Liveness checks and
>    positive controls that include *recovered* events, gated before every
>    paid run.
> 3. **Never consume output without asserting the producer succeeded** —
>    including your own ledger, your own fixtures, your own event stream.
> 4. **First contact with a clean host is a validation step, not a
>    deployment step.** Six of our defects could not exist on the machine
>    that wrote them.
> 5. **One environment.** The agent executes where it is measured, enforced
>    by architecture, not by discipline. Our instrument had this seam for
>    its probes and not for its agent; that asymmetry cost the most.

The taxonomy is the argument that these rules generalise: the mechanisms are
properties of subprocess-and-PATH, containers, parsers, and git — components
every harness has — not of our code. Class A is OpenHands' issues #4235 and
#7044 (environment construction against SWE-bench images; missing `/testbed`
and conda profile) [6]. Class D is SWE-bench's own grader consuming forgeable
stdout [5]. The insufficiency UTBoost documents — 345 leaderboard patches
mis-graded resolved, affecting a quarter of Verified's entries [4] — is
Class B one level up: the oracle's holes rendered as passing measurements.
Ours differ only in having been found by the people who wrote them.

---

## 7. Related work

SWE-bench [1] and its Verified subset [2] are the substrate. TDAD [3]
measures regressions on Verified from the *final* patch — the design §2
argues under-counts by construction, and the concurrent point of comparison
for event-level measurement. UTBoost [4] audits the benchmark's oracle
itself and finds it insufficient at leaderboard-moving scale. The SWE-bench
grader's stdout-trust defect [5] and OpenHands' environment-construction
issues [6] are the in-the-wild instances of our classes D and A. SWE-bench
Live [7] addresses contamination with rolling instances and carries a
median held-out oracle ~37× larger than Verified's — relevant to any
event-level instrument, since the oracle is the net. Broader
evaluation-infrastructure work [8] catalogues agent-eval irreproducibility;
our contribution is narrower and mechanistic: what breaks *inside* one
instrument, and which invariants stop each class.

## 8. Limitations

Everything here is exploratory; no claim about recovery policy is made, and
the harness's benchmark resolve rate is **unmeasured** — stated plainly
because the alternative is implying competence we cannot show. The
held-out oracle observes roughly the gold test-patch's blast radius (97.5%
of it in one file for this benchmark), so "silent regression" here means
silent *within the tests adjacent to the edit* — a declared lower bound.
The catalogue is a census of one team's instrument, organised into a
taxonomy whose external instances are evidence of reach, not a survey. The
fix for A6 is built and golden-checked on real images (§5.1), but no
benchmark measurement produced through it is reported here — the first
valid numbers postdate this submission by design.

---

### References

[1] Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. arXiv:2310.06770.
[2] OpenAI. *Introducing SWE-bench Verified.* 2024.
[3] *TDAD.* arXiv:2603.17973.
[4] Yu et al. *UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench.* ACL 2025. arXiv:2506.09289.
[5] SWE-bench issue #601: *Test Result Hijacking via Stdout Forging in Evaluation Harness.* github.com/swe-bench/SWE-bench/issues/601.
[6] OpenHands issues #4235, #7044: environment errors in SWE-bench instance evaluation.
[7] Zhang et al. *SWE-bench Goes Live!* arXiv:2505.23419.
[8] *Holistic Agent Leaderboard.* arXiv:2510.11977.
