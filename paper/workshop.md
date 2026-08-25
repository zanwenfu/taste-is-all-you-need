# The Instrument Is the Experiment: Silent Failure Modes in Measuring Agent Harnesses

*AgenticOS @ NeurIPS 2026 — regular paper, draft. ANONYMISE BEFORE SUBMISSION.*

---

## Abstract

An OS layer for agentic AI will be built out of design choices — how state is
checkpointed, what happens when a step fails verification, when work is rolled
back — and those choices can only be compared if they can be measured. We set
out to measure one of them: whether rolling back to a verified checkpoint
leaves less previously-working behaviour broken than repairing in place. We
built the instrument the comparison needs, validated it against cases with
known answers, and ran it on SWE-bench Verified.

What we report is not that comparison. It is that **the instrument failed
fifteen times, and thirteen of those failures produced a plausible number
rather than an error.** Twelve were invisible to a 533-test suite that was
green throughout. Four were invisible on any developer machine by
construction, because they depend on the *absence* of ambient configuration
that a developer machine has. The most expensive single failure would have
reported "no contamination anywhere" on a substrate where 46% of instances
could not run the probe at all.

We contribute: (i) a measurement protocol for regressions in long-horizon
agent runs that observes a timeline rather than a final state, (ii) a
validation gate that fails loudly when the instrument is dead — which we show
is *not* the default behaviour, and (iii) a catalogue of the fifteen failures
with mechanisms and fixes. We argue the catalogue is the more useful artifact:
every failure mode we found is available to anyone building observability into
an agent OS, and most of them announce themselves as clean results.

---

## 1. Introduction

The agentic-systems community is converging on the idea of an OS layer:
common abstractions for memory, scheduling, checkpointing and recovery
[cite workshop CFP]. Abstractions are chosen, and choosing between them
requires evidence. This paper is about how much harder producing that evidence
is than it looks.

Our motivating question is narrow and concrete. When a step in a long-horizon
agent run fails its verification, the harness must do something. Two policies
dominate deployed systems: **repair in place** (hand the model its own broken
tree and ask it to fix forward) and **roll back and retry** (reset to the last
verified checkpoint and try again with feedback). The second is more
expensive — it discards work and pays full uncached rates on the restart — and
the argument for it is that it leaves less collateral damage behind.

That claim is testable. It requires knowing, for each run, when previously
working behaviour stopped working. It turns out that knowing this is where
the difficulty lives.

**Contribution.** We report an engineering result rather than a scientific
one. Building an instrument that can answer "when did this break?" for an
agent run is an unsolved problem whose failures are systematically silent: a
broken instrument and a clean run produce the same number. We document
fifteen distinct instances of this, give the mechanism for each, and describe
the two design decisions that caught most of them.

---

## 2. What has to be measured

**A silent regression** is a test that passed at some observation of a run and
fails at a later one, with nothing in the harness reporting it. Three
properties make this hard to measure and are the source of every failure in
§4.

**It is an event, not a state.** A regression that is introduced and then
repaired leaves no trace at the end of a run. Any instrument that inspects the
final artifact — which is what published work does, including concurrent work
measuring regressions on this same benchmark [TDAD, arXiv:2603.17973] —
records zero. We observed exactly this case: eight assertions broke at
observation 3 of a real run and were passing again at observation 4, because
the harness's own rollback erased them. A final-state measurement reports
nothing happened.

**Its absence and the instrument's death are the same observation.** Zero
regression events is what a clean run produces. It is also what a probe that
cannot execute produces, what a parser that matched nothing produces, and what
an empty timeline produces. This is the paper's central practical claim, and
§4 is fifteen instances of it.

**Detection must be attributed, not co-located.** "The harness failed
something while the regression was open" is not evidence the harness noticed
the regression. It systematically credits whichever policy fails most often
with the best detection — which is the opposite of the quantity under study.

---

## 3. The instrument

**Observational checkpointing.** Every mutating tool call writes a commit on a
ref the agent cannot enumerate, built from a private index so the agent's own
`git status` and `git diff` are byte-identical with the instrument on and off.
This gives every policy the same observation granularity, which is the only
way detection latency means the same thing across them.

**Exhaustive replay, not bisection.** Held-out probes are replayed at *every*
observation. An earlier version binary-searched for the onset, assuming the
verdict sequence is monotone. That assumption is not merely approximate here —
it is violated *systematically and differentially by policy*, because
non-monotonicity is the treatment. Bisection recorded only regressions that
survived to the end of a run, which are exactly the ones the recovering policy
*failed* to fix. The policy whose entire claim is recovery would have shown
the fewest regressions and no measurable recovery rate.

**Attribution by coverage.** A harness failure is linked to a regression only
when some failing harness test and the broken graded test both exercise a file
the agent changed at that observation. All three terms are load-bearing; drop
the third and any two tests sharing a utility module link forever. Both the
attributed and the co-occurrence variants are always reported, with the latter
labelled as what it is — an over-count of detection, and therefore an
under-count of silence.

**Gate 0.** Five checks with thresholds fixed in advance: negative control,
positive control (including *recovered* regressions), flake screen, unknown
rate, and **baseline liveness**. The last exists because of §2's second
property, and §4 shows it is the check that matters.

---

## 4. Fifteen ways to measure nothing

Each row is a real defect found in this work. **Silent** means it produced a
plausible number rather than an error. **Green** means the test suite passed
while it was present.

| # | Defect | What it reported | Silent | Green |
|---|---|---|---|---|
| 1 | Bisection assumed monotone verdicts | recovered regressions invisible; recovering policy looks cleanest | ✓ | ✓ |
| 2 | Observation staged into the real git index | agent's own `git diff` returned empty | ✓ | ✓ |
| 3 | Probe ran `pytest` with django's unittest ids | 46% of the benchmark contributes zero episodes | ✓ | ✓ |
| 4 | Output markers used `:` (a shell no-op printing nothing) | a perfect 13/13 run scored as 13 errors | ✓ | ✓ |
| 5 | Results on stderr, markers on stdout | django results fall outside the parsed slice | ✓ | ✓ |
| 6 | Probe diffed against a commit deliberately absent | every graded test a hole → "0 regressions" | ✓ | ✓ |
| 7 | Cached mirror accepted with zero commits | later instances die far from the cause | ✓ | ✓ |
| 8 | Checkpoint card field never populated | audit trail reported 0 tool errors, always | ✓ | ✓ |
| 9 | One malformed planner response killed the run | 33% of runs lost, scored as *task* failures | ✗ | ✓ |
| 10 | Dependency install counted as agent work | 3,640 of 3,641 changed files; attribution term goes vacuous | ✓ | ✓ |
| 11 | Executed config rebuilt, ledger recorded the other one | manifest describes a run that never happened | ✓ | ✓ |
| 12 | Shadow chain inherited the machine's git identity | **empty timeline on any clean machine** | ✓ | ✓ |
| 13 | Gate 0's probe invoked a bare `python` | every probe exit 127 on a clean machine | ✗ | ✓ |
| 14 | Unpinned SDK resolved a different major version | every model call failed; two machines ran different code | ✗ | ✓ |
| 15 | Isolation removed loopback, not just the internet | 23.5% of the oracle dead; one instance 812 tests | ✓ | ✓ |
| 16 | Verification commands invoked a bare `pytest` | **15 tests fail on a clean host, as harness defects** | ✗ | — |
| 17 | Benchmark scorer invoked a bare `python` | `fractional_score` returns 0.0 — a score from a missing interpreter | ✓ | ✓ |
| 18 | Git handles never released | sweeps die of `Too many open files` **after ~400 cells** | ✓ | ✓ |

Fourteen of eighteen were silent. Fifteen were present while the suite was
green — and #16 is the suite itself.

### 4.2 Three that arrived together, and what connects them

The last three were found in one sitting and share a mechanism worth naming.

**#16 is the measuring stick.** Our fixtures verified agent work by running
`pytest -q`. Bare, that resolves against `PATH`, which does not contain the
virtualenv's `bin` when the suite runs as `python -m pytest` — the normal way
anywhere nobody typed `activate`. On the machine this was written on, an
ambient pytest happened to be installed, and the suite was green. On a clean
host **fifteen tests failed, including every test of the rollback thesis** —
and they failed *reading as harness defects*: "rollback did not recover", "the
merge gate rejected independent work". The suite that certified every other
fix was certifying nothing on any machine but one.

**#17 is the same defect in the path that produces reported numbers.** The
Commit0 scorer shelled out to `python -m pytest`; a clean Ubuntu ships
`python3` and no `python`. The command exits 127 and the scorer returns
`0.0` — a benchmark score of zero manufactured by a missing interpreter and
arithmetically indistinguishable from an agent that implemented nothing.

**#18 was invisible until an assertion was added.** A GitPython `Repo` holds
a `cat-file --batch` process pair and mmaps every pack it touches, and nothing
closed them. One leak is unnoticeable; at the default 1024-descriptor limit,
the four-hundredth cell of a sweep starts dying of `Too many open files`. The
sweep driver records those cells as `error` and drops them from the
denominator — so the loss lands **entirely on the back half of a sweep**,
ordered rather than random. A confirmatory run over 485 instances would have
silently measured a biased prefix of its own frame.

It surfaced only because of a change to a *test fixture*: the fixture used a
sweep's output without first asserting the sweep succeeded, so the failure
presented as an unrelated `IsADirectoryError` inside `pathlib` while the
actual traceback sat unread in the cell's `error` field. Adding two lines —
assert the status, assert the sidecar exists — turned a mystery into the
message `OSError: [Errno 24] Too many open files`.

That is the generalisable rule, and it is cheap: **never consume a
measurement's output without first asserting the measurement succeeded.** Every
silent failure in this table is an instance of some component doing exactly
that.

**Four were unfindable on a developer machine.** #12, #13 and #14 depend on
the *absence* of ambient configuration — a git identity, a `python` alias, a
pinned dependency set — and a developer machine has all three. They appeared
within an hour of first running on a clean cloud host. A clean host is what a
reproduction is.

### 4.1 The two decisions that caught most of them

**Infrastructure failure is a missing observation, never a test failure.** A
container that will not start, a patch that will not apply, a timeout, an
unparseable log, a test the runner never mentioned — all yield `error`, never
`fail`. This single rule converted #4, #5, #6 and #15 from *fabricated
regressions* into *visible holes*. Without it each would have reported
contamination at every observation of the affected instances.

**Zero events must be distinguishable from a dead instrument.** Gate 0's
`baseline_liveness` check asserts that a clean run's probes actually answer
`pass`. Before it existed, forcing every probe to error made the negative
control return **1.000 PASS** — a perfect score for an instrument that could
not run anything. On first contact with the clean cloud host, the gate failed
loudly (`baseline liveness 0.000`, negative control reporting `dead` for every
sample) on a defect that would otherwise have been certified as a clean result
on the machine we were about to spend real money on.

---

## 5. What the corrected instrument measures

*[PENDING: corrected 40-instance pilot. Do not write until the numbers land.]*

Reported as exploratory throughout. These runs use a development slice
declared in advance and excluded permanently from any confirmatory frame,
because the instrument was debugged against them.

---

## 6. What this means for an agent OS

If the field intends to standardise an OS layer, observability is one of the
primitives, and our experience suggests three things about it.

**Instrument validation belongs in the layer, not in each user's harness.**
Every failure in §4 was in the measurement path rather than the agent path.
The agent OS worked: plans decomposed, workers executed, the trap handler
diagnosed faults from tree state alone and escalated to `halt` when it
recognised no progress. What repeatedly did not work was the code that watched
it.

**Silence must be typed.** The single most valuable rule we adopted was
refusing to let "could not measure" render as "measured nothing". An
observability API for agents should make that distinction unrepresentable
rather than conventional.

**Reproducibility failures are the same failure.** #12, #14 and #15 are all
"works on the machine that built it". For a layer meant to make agent runs
sharable, an instrument whose output depends on the host's git config is not a
minor defect.

---

## 7. Limitations

- Everything here is **exploratory**. No confirmatory contrast between
  recovery policies is claimed; the sweep it would require is registered but
  not run.
- The pilots are small and drawn from a development slice.
- Our harness's own verification is stricter than the benchmark's grading, so
  most runs do not complete. Completion rate and regression rate are not
  independent, and that interaction is unmeasured.
- `PASS_TO_PASS` is an edge-biased lower bound: 97.5% of it sits in the file
  the gold test patch edits, and 91.2% of instances have their entire oracle in
  a single test file.
- The bug catalogue is a census of what *we* hit, not a taxonomy. Its
  generality is an argument, not a measurement.
