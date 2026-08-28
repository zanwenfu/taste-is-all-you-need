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

614 tests passing, lint clean, on the experiment host (counted, not remembered, at this revision). Every catalogue fix
ships with a test verified to fail when the fix is reverted.

## Formerly in flight — now complete

The equal-N GPT-5.6 run (pilot 40d) and the recovery-policy contrast both landed 2026-08-27; see their sections below.

## Pilot 40d — GPT-5.6 stack, equal-N (exploratory, dev slice, 2026-08-27)

| fact | value | source |
|---|---|---|
| cells attempted | 37 of 40 (breaker stopped the sympy tail after 6 consecutive zero-progress failures; 3 cells skipped; the ledger's 38th row is the abort marker) | ledger /root/pilot40d |
| statuses | 15 completed, 18 failed, 2 infra, 2 error (bug-29 symlink sync, both sphinx) | ledger |
| resolve | 13/35 graded = 37.1% | official grader |
| spend | $8.14 (Claude same slice: $34.24) | ledger |
| observations | 259 | evidence |
| raw episodes / declared events | 162 / 140 | evidence |
| distinct incidents (onsets) | 11 | computed from evidence |
| bearing runs | 6/37 = 16.2%, CI [6.2%, 32.0%] | evidence |
| matched-slice run-level contrast | 0/40 (Claude) vs 6/37 (GPT): Fisher one-sided p = 0.0098 | computed |
| fully-silent episodes (no co-occurring failure) | 48 of 162 | evidence sidecars |
| bearing instances | django-11276, django-15128, matplotlib-26291, seaborn-3069, pylint-6386, pytest-6197 | evidence |
| oracle holes | 20 across 2 instances (requests 4 by network design; sklearn 16) | evidence |

## Bug 29 (post-census; described with the golden-gate catches)

Sync pull died on symlinked test fixtures (sphinx); two paid cells lost as
typed errors. Fixed with visible-skip semantics (`router.skipped`),
revert-verified test. Outside the 28-row census, which closed at submission.

## The undercount (all bearing runs, both valid GPT sweeps; $0, from sidecars)

184 declared events across 8 bearing runs; **1** visible as a final-state
P2P failure (net of typed oracle holes). Final-state capture rate: 0.5%.
Per-run rows: flask-5014 41→0, pytest-6197 3→0 (twice), django-11276 2→0,
django-15128 48→0, matplotlib-26291 7→0, seaborn-3069 73→0,
pylint-6386 7→1. Source: /tmp/undercount.py over evidence sidecars.

## The recovery-policy contrast (dev slice, GPT-5.6 stack, 2026-08-27; analysis committed before unblinding)

| fact | value | source |
|---|---|---|
| arms | rollback (pilot40d), repair-in-place (contrast40_A2), no recovery (contrast40_A0); 40 instances each | ledgers |
| resolve | rollback 13/35 = 37.1%; repair 24/40 = 60.0%; no recovery 22/40 = 55.0% (after re-grading the 7 patch-killed cells) | official grader + rescore |
| spend | $8.14 / $14.73 / $4.40 | ledgers |
| cells with final-state contamination | rollback 1; repair 5; no recovery 9 | grades net of typed holes |
| PRIMARY, no recovery vs rollback (paired sign) | worse 9 / better 1 / ties 25 → exact p = 0.0215 | scripts/contrast.py |
| repair vs rollback (paired sign) | worse 5 / better 1 / ties 29 → p = 0.2188 | scripts/contrast.py |
| CO-PRIMARY onset exposure, no recovery vs rollback | more 2 / fewer 5 / ties 28 → p = 0.4531 | scripts/contrast.py |
| paired resolve, McNemar exact | vs repair: rollback-only 1, repair-only 9 → p = 0.0215; vs no recovery: 3 vs 9 → p = 0.1460 | /tmp/mcnemar2.py on rescored sidecars |
| first unblinding (pre-fix) | primary p = 0.375 — 7 cells dropped as ungradable (bug 30); pointer in §5.4, full disclosure in Appendix B | log |
| exhibit | django-13012 no-recovery final tree: `$(cat django/db/models/expressions.py)` as line 1; 137 graded tests missing; rollback arm graded 137/137 | direct container diagnostic |

## Bugs after the census (all fixed, revert-verified): 29 sync symlinks; 30 patch-killed suite scored as hole; 31 rescore never graded; 32 harness artifacts leaked into the prediction.

## Submission artifact (2026-08-27)

paper/latex/main.pdf — 7 pages: 6 technical + references. Built with the
official NeurIPS 2026 style via paper/latex/md2tex.py; 0 overfull boxes.
Anonymised (converter drops the draft header; sweep clean).


## Catalogue-row provenance (numbers cited in §4–§5 prose; all in docs/research_log.md)

| figure in the draft | value | source |
|---|---|---|
| "after nineteen fixes" | bug 20 was found after bugs 1–19 were closed | log, session 13 |
| re-scoring "8 and 6 episodes exactly" | pilot-10 archived trees re-scored under the pilot-40 instrument | log, session 12 |
| B1 "13/13 run scored as 13 errors" | the `:` marker defect on a perfect pytest run | log, session 11 (bug 2) |
| B6 "23.5% of the oracle dead; 812 tests" | network_disabled=True; matplotlib-26113 `{'error': 812}` → `{'pass': 812}` | log, session 12 (bug 15) |
| C2 "λ̂ off by up to 2.7×" | declared-unit collapse 0.80 → 0.30 on the pilot | log, session 12 |
| D2 "33% of runs lost" | planner JSON-string steps crashing runs | log, session 11 (bug 7) |
| D3 "3,640 of 3,641 changed files" | npm install counted as agent work on a Harbor task | log, session 11 (bug 8) |
| D6 "after ~400 cells" | GitPython descriptor leak against the default 1024 limit | log, session 12 (bug 18) |
| D10 "78-event storm" | first GPT-5.6 calibration, seaborn, breaker-stopped at $1.12 | log, session 15 (bug 28) |
| §3 (golden gate) "~3 points forever" | grade under `none` network: requests 126/130 vs 130/130 under bridge | log, session 14 |
| §3 (golden gate) "three hardest repo families" | golden checks: requests (pytest), django (runtests), matplotlib (compiled ext.) | log, session 14 |

## Why rollback runs failed (pilot 40d, rollback arm; 2026-08-27)

| fact | value | source |
|---|---|---|
| failed runs | 18 of 37 attempted (the 38th ledger row is the abort marker) | ledger |
| failed at the first step | 15 of 18 | ledger steps_passed |
| rejecting check kinds | 8 pytest/runtests, 3 `python -c`, 7 other — all planner-written | ledger failure_reason |
| false rejections | 9 of 18 instances resolved under no-recovery or repair-in-place | evidence, three arms |
| spend on failed rollback runs | $5.22 | ledger |

## Regression-gated rollback (arm A3reg; dev slice; GPT-5.6; 2026-08-27)

| fact | value | source |
|---|---|---|
| cells | 40: 35 completed, 3 failed, 1 infra (xarray-3305, provider 400 in every arm), 1 budget | ledger /root/contrast40_A3reg |
| resolve | 26/40 = 65.0% | official grader |
| spend | $10.51 (+$2.37 over plain rollback) | ledger |
| contaminated final trees | 0 | grades net of holes |
| events / bearing | 70 declared, 5/40 bearing, CI [4.2%, 26.8%] | evidence |
| paired resolve vs plain rollback | both 12, neither 12, rollback-only 1, gated-only 10 → McNemar p = 0.0117 | /tmp/mcnemar3.py |
| paired contamination vs plain rollback | gated better on 1, worse on 0, ties 34 | scripts/contrast.py |
| co-primary onset exposure | more 3 / fewer 5 / ties 27 → p = 0.73 | scripts/contrast.py |
| the three gated failures | genuine regressions refused: 8 tests, 2 tests, suite killed | ledger failure_reason |

## Corrections from the second fresh referee (2026-08-27)

| item | value | source |
|---|---|---|
| 40d cells | 37 attempted of 40 (3 skipped by breaker; the 38th ledger row is the abort marker); 35 graded (2 errors) | ledger |
| resolve on the common denominator | rollback 13/40; gated 26/40; repair 24/40; no recovery 22/40 | grades |
| observation density, 40-run sweeps | Claude 227/40 = 5.7; GPT 259/37 = 7.0 per run; ratio 1.2 (1.4 was the calibration) | evidence |
| pooled undercount denominator | 47 GPT rollback runs (10 calibration + 37 sweep); 8 bearing; run-weighted 7 of 8 clean at the end | evidence |
| storms | seaborn 73, django-15128 48, flask 41 = 162 of 184 events (88%) | evidence |
| gate status, both stacks | bearing 0% (Claude) and 20%/16.2% (GPT) < 25%: conjunction FAILED for both; no confirmatory arm | prereg + evidence |
| repair-vs-rollback co-primary | onset exposure more 4 / fewer 5 / ties 26 → p = 1.0 | scripts/contrast.py |
| per-cell cap | $4 work-cost | driver flags |
| A4 | 15 tests failed on the clean host | log, session 12 |
| the 7 ungraded cells | 5 no-recovery + 2 repair | regrade log |
| pytest-6197 onsets | 3, 5, 7 | evidence |
| django share | of Verified 46.2%; of the 40-slice: see draft (computed from the slice) | ledger |
| provenance pointers | golden-check facts now cited in §3; the unblinding disclosure in §5.4 | this file |


## Replication of plain rollback (GPT-5.6, same slice; 2026-08-27)

| fact | run 1 (pilot 40d) | run 2 (replicate40_A3) | source |
|---|---|---|---|
| cells attempted | 37 of 40 (breaker) | 36 of 40 (breaker; 37th row is the abort marker) | ledgers |
| resolve | 13/35 graded (37.1%) | 13/36 graded (36.1%) | grades |
| paired resolve (34 shared graded) | 13 | 13 — both 11, neither 19, run1-only 2, run2-only 2 | evidence |
| bearing runs | 6/37 = 16.2% | 6/37 = 16.2% | evidence |
| declared events / raw | 140 / 162 | 116 / 127 | evidence |
| final-state contamination, paired | run 2 better on 1, worse on 0, ties 33 (p = 1.0) | | scripts/contrast.py |
| onset exposure, paired | run 2 more on 2, fewer on 3, ties 29 (p = 1.0) | | scripts/contrast.py |
| spend | $8.14 | $6.87 | ledgers |

## Answers to the external review (2026-08-28)

| item | value | source |
|---|---|---|
| storm root causes | seaborn-3069: +6/−1 lines in seaborn/_core/plot.py, parses, no shell idiom (95 raw events); django-15128: +9/−4 in django/db/models/sql/query.py, parses (48); flask-5014: +46/−262 in src/flask/blueprints.py, parses (54); all worker stops end_turn (no output cap) | /tmp/reviewer_numbers.py over shadow timelines |
| spend while a regression was open | $0.63 of $2.33 across the 8 bearing runs = 27% (range 1%–66% per run) | shadow cost stamps × episode intervals |
| gate oracle vs P2P | baseline passing / \|PASS_TO_PASS\| median 1.00 (min 0.67, max 1.56) over 40 gated cells | gate.baseline events |
| model API strings | claude-opus-4-7 / claude-sonnet-4-6; gpt-5.6-sol / gpt-5.6-terra | run manifests; taste/pricing.py |
| Figure 4 denominators | resolve now shown as x/40 for every arm (13, 26, 24, 22) | figs.py |
| Claude vs GPT trade | 13/40 resolved each; $34.24 vs $8.14; 0 vs 140 events | ledgers |
| split-oracle gate arm (A3reg2) | gate watches a deterministic half of each instance's test files (sha1 of path, even); grader scored on the held-out half via `grade_failed` in the sidecar | code + sweep in flight |

## Replay cost (timed re-scores, 2026-08-28)

| cell | observations | wall | per observation | source |
|---|---|---|---|---|
| pytest-dev__pytest-6197 | 9 | 69.5 s | 7.7 s | /root/replay_timing.log |
| django__django-15128 | 5 | 19.9 s | 4.0 s | same |
| mwaskom__seaborn-3069 | 9 | 32.6 s | invalid — collided with the concurrent split sweep on the same instance (defect 34); all 94 tests read as holes | same |
| 40-instance sweep projection | 259 obs × ~6 s ≈ 26 CPU-min | | computed |
| 500 instances × 100 edits | ≈ 80 CPU-hours | | computed |

## Split-oracle gate, file-level pilot (superseded; 2026-08-28)

| quantity | value | source |
|---|---|---|
| cells completed before the arm was stopped | 26 graded (11 resolved), 8+ infra (empty watched half) | /root/contrast40_A3reg2_filesplit |
| P2P ids: watched by the gate / held out (files the gate never ran) | 1847 / 2187 | scripts/heldout.py |
| cells with a held-out failure in the final tree | 0 of 26 | same |
| cells with a watched failure in the final tree | 0 of 26 | same |
| why superseded | file-level split leaves single-file instances with nothing to watch; id-level split replaces it | regression_gate.py docstring |

## Split-oracle gate, id-level (A3reg2, GPT-5.6, 2026-08-28) — the held-out answer to the circularity objection

| quantity | value | source |
|---|---|---|
| cells | 40: 38 completed, 1 budget (sphinx-8638, $2.25), 1 infra (sphinx-7985, $0) ; wall 6883 s ; spend $9.40 | /root/contrast40_A3reg2.log |
| resolve | 28/40 = 70.0% (all 40 graded; unfinished cells count as unresolved) | pilotstats |
| gate reads | sha1-even half of previously-passing ids: 2278 watched, 2585 held out (never read by the gate) | scripts/heldout.py |
| cells with a held-out failure in the final tree | 0 of 40 on the instrument's endpoint. The official grade flags scikit-learn-14710 (11 held-out + 5 watched ids, all `test_init_parameters_validation[...]` cases): those 16 ids are stored truncated at the first space in the dataset and the current upstream parser keys the full id, so they match nothing in any arm (47/63 in all six arms). Direct check: with the split-gate final patch applied in the raw image, all 32 validation cases pass. Defect 35 (benchmark mechanism, A.3). | scripts/heldout.py; /tmp/sk14710_split.patch |
| instrument final-state contamination | 0 cells (the 16 ids are baseline-dead in the agent's tree: they exist only under the hidden test patch) | pilotstats / contrast.py ties 40 |
| paired resolve vs plain rollback (35 shared) | both 11, rollback-only 2, split-only 13; McNemar exact p = 0.0074 | inline McNemar |
| paired resolve vs full-oracle gate (40 shared) | both 23, full-only 3, split-only 5; p = 0.73 | same |
| paired contamination vs plain rollback | split better on 1, worse on 0, ties 34 (sign p = 1.0) | contrast.py |
| onset exposure vs plain rollback | more on 2, fewer on 6, ties 27; p = 0.29 | contrast.py |
| events | 225 observations, 14 declared, 3 bearing runs (7.5%) | pilotstats |

## Defect 35 — dataset/parser id drift (found 2026-08-28 via the split-gate arm)

| quantity | value | source |
|---|---|---|
| mechanism | SWE-bench dataset ids for parametrised cases were produced by the v4.1.0 parser (`test_case[1]`, truncated at the first space); upstream `main` `parse_log_pytest_v2` keys `" ".join(test_case[1:])`; our port follows `main` (delta 2 in swebench_log.py) | upstream python.py at main vs v4.1.0, fetched to /tmp on the box |
| exposure in the 40-instance slice | 6 instances carry 81 truncated ids; only scikit-learn-14710 (16 ids) is on the v2 parser; pytest/requests/matplotlib parsers truncate and match | census in /tmp/gradecheck.py |
| consequence | scikit-learn-14710 grades unresolved in every arm (F2P 1/1, P2P 47/63) regardless of patch; the instrument marks the 16 ids baseline-dead (excluded) so contamination endpoints are unaffected | grade summaries across six roots |
| decision | keep mirroring `main` (the stated grading contract); report the cell as unresolved everywhere; catalogue the mechanism in A.3 rather than rescue the cell | this session |

## Trajectory length (mutating tool calls per run = observations), 2026-08-28

| arm | runs | mean | median | IQR | max |
|---|---|---|---|---|---|
| Claude rollback (pilot40c) | 40 | 5.7 | 5 | 4–8 | 10 |
| GPT rollback (pilot40d) | 35 | 7.4 | 7 | 4–9 | 13 |
| gated (A3reg) | 40 | 6.3 | 6 | 4–9 | 12 |
| split-gated (A3reg2) | 40 | 5.6 | 4 | 4–7 | 12 |
| repair-in-place (A2) | 40 | 6.9 | 6 | 5–9 | 19 |
| no recovery (A0) | 40 | 4.7 | 4 | 4–5 | 13 |
| all six arms pooled | 235 | — | 5 | 4–8 | 19 (94% of runs ≤ 10) |

Source: /tmp/obsdist.py over evidence sidecars (`observations`). Public-scaffold trajectories on Verified run to tens or hundreds of actions; the §3 projection assumes 100 edits per instance.

## Cross-stack Fisher sensitivity (one-sided, Claude 0/40 bearing vs GPT k/37)

| k | p |
|---|---|
| 6 (observed) | 0.0098 |
| 5 | 0.022 |
| 4 | 0.049 |
| 3 | 0.106 |

Source: /tmp/obsdist.py. Reassigning three of the six bearing runs removes significance at 0.05.

## Deployed-gate cost: full repository suite in the pinned image (2026-08-28)

| instance | command | wall | outcome | source |
|---|---|---|---|---|
| django__django-11133 | `tests/runtests.py --parallel=4 -v0` | 111 s | FAILED: failures=10, errors=87, skipped=902, xfail=4 (baseline-dead in the image) | /tmp/fullsuite.log on the box |
| pytest-dev__pytest-6197 | `pytest testing -q` | 3 s | 2 collection errors — the image cannot run its own full suite unscoped; not a usable timing | manual run |
| scoped gate run (member files) / replay | — | 4–8 s per observation | — | §3 replay timing |
