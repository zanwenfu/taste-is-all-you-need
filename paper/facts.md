# Verified numbers, with provenance

Nothing enters the draft unless it appears here with a source. This file is
the defence against the failure mode the paper is about: a number that looks
measured and is not. Rebuilt 2026-08-27; the draft's every figure was
re-checked against this revision.

## The frame (SWE-bench Verified, local pin sha256 c008a795…)

| fact | value | source |
|---|---|---|
| instances | 500 | dataset |
| eligible (non-empty PASS_TO_PASS) | 489 | `eligible()` |
| after dev-contamination exclusions | 485 | prereg E4b |
| P2P tests, whole dataset | 60,142 | dataset |
| P2P median per instance | 50.5 | measured |
| repo share, django | 231/500 = 46.2% | dataset |

## The catalogue

**28 defects** (rows A1–A6, B1–B8, C1–C4, D1–D10). 23 silent; 27 present
under a green suite (A4 is the suite); 6 clean-host-only. Census closed at
submission; the golden gate's three pre-spend catches (§5.2) are outside it.

## Early pilots (invalid frame — bug A6; retained for the capstone only)

80 runs, **$99.07**, 26/28 zero-step runs traced to A6. All withdrawn claims
listed in §5. (The "$110" figure that circulated earlier included the
10-instance pilot's $11.48 and is retired.)

## Pilot 40c — Claude stack, valid pipeline (exploratory, dev slice)

| fact | value | source |
|---|---|---|
| cells | 40, all routed + graded | ledger /root/pilot40c |
| resolve | 13/40 = 32.5%, CI [18.6%, 49.1%] | official grader per cell |
| spend | $34.24 | ledger |
| observations / replays | 227, replays == observations on every cell | evidence |
| events (declared unit) | 0; bearing 0/40, CI [0%, 8.8%] | evidence |
| same-regime canary | 1 bearing run, 3 declared events (requests-5414) | /root/canary evidence |
| rescore control | canary reproduces 8 raw / 3 declared exactly under the 40c instrument | rescore run |
| observation density | 5.7 / run | evidence |

## GPT-5.6 calibration — same 10-instance prefix (exploratory, dev slice)

| fact | value | source |
|---|---|---|
| cells | 10 (re-run; first attempt destroyed by D10, breaker-stopped at $1.12, excluded as infrastructure) | ledger /root/oai10b + /root/oai10 |
| resolve | 5/10 (Claude on same 10: 6/10) | official grader |
| spend | $1.77 (Claude same 10: $8.83) | ledger |
| raw episodes | 57 | evidence |
| declared events | 44 | evidence |
| distinct incidents (onsets) | 4 | evidence |
| bearing runs | 2/10, CI [2.5%, 55.6%] | evidence |
| run-level contrast vs Claude | Fisher exact p ≈ 0.24 (0/10 vs 2/10) | computed |
| observation density | 8.2 / run; events/observation 0.70 vs 0.00 | evidence |
| detection split of 57 | 39 attributed-detected-then-erased; 16 co-occurred unattributed; 2 fully silent (both inside the resolved run) | evidence sidecars |
| final-state visibility | 0 of 57 (all recovered; bearing instances grade 59/59 and 145/145) | official grader |
| capped turns in event logs | none (D10 guard active; revert-verified test) | event logs |
| exhibit: flask-5014 | 54 test functions broken at one observation, all erased by rollback, final grade clean; Claude resolved the same instance with zero events | evidence |
| exhibit: pytest-6197 | resolved (F2P 2/2, P2P 145/145) with 3 regressions en route — 2 never noticed by the harness | evidence |

## Gate status under the pre-declared rules

Stack two: λ̂ clears its threshold; **bearing 20% < 25% — the declared
conjunction FAILS**. Consequence: not a confirmatory pass; the gates are
re-declared regime-conditional in prereg amendment 1 (drafted before the
contrast arms were unblinded).

## SWE-bench-Live (adapter validated; no measurements yet)

| fact | value | source |
|---|---|---|
| lite split | 300 instances, 300/300 pytest parser, 292 file-scopable | fetched dataset |
| P2P median (lite) | 1,711 (~34× Verified) | fetched dataset |
| golden check | gold F2P 26/26, zero new P2P failures; null False | livegolden run |
| time-rotted oracle | 34/1,220 baseline tests fail in the raw image from calendar drift (cfn-lint-3798) | raw-image reproduction |

## Test suite

604 tests passing, lint clean, on the experiment host. Every catalogue fix
ships with a test verified to fail when the fix is reverted.

## In flight at submission time

Equal-N GPT-5.6 run (40 instances) and the pre-declared recovery-policy
contrast (repair-in-place and no-recovery arms), with the paired analysis
committed before unblinding (scripts/contrast.py).
