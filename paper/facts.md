# Verified numbers, with provenance

Nothing enters a draft unless it appears here with a source. Written as a
defence against the failure mode this paper is *about*: a number that looks
measured and is not.

## The frame (SWE-bench Verified, local pin sha256 c008a795…)

| fact | value | source |
|---|---|---|
| instances | 500 | dataset |
| eligible (non-empty PASS_TO_PASS) | 489 | `eligible()` |
| after removing dev-contaminated | 485 | prereg E4b |
| gold patch churn, median | 7 lines | measured |
| gold patch files, median | 1 | measured |
| single-file share | 85.8% | measured |
| \|PASS_TO_PASS\| median | 50.5 | measured |
| instances rated ≥1 hour | 45 (9%) | dataset `difficulty` |
| repo share, django | 231/500 = 46.2% | dataset |
| runner split | 231 runtests.py / 150 pytest / 75 bin/test / 44 tox | measured, 500/500 build |

## Oracle identifier grammars (all 60,142 P2P ids)

| grammar | share |
|---|---|
| pytest node id | 61.5% |
| unittest label | 24.7% |
| bare function name (sympy) | 6.5% |
| unittest label + method (django 4.2+) | 1.3% |
| **not a test identifier at all** | **6.1%** |

93.9% (56,499/60,142) parse. django 81.1%; the shortfall is docstrings printed
in place of test names.

## Network isolation

| container | P2P pass | fail | instance |
|---|---|---|---|
| network severed | 44 | 35 | psf__requests-1724 |
| network enabled | 78 | 1 | same |

Separately, `network_disabled=True` (removing loopback) vs `network_mode=none`
(keeping it) on matplotlib-26113: `{'error': 812}` → `{'pass': 812}`.

## Clean-host defects (found by first contact with a fresh cloud box)

| defect | effect on a clean host | effect on the dev machine |
|---|---|---|
| gate 0 probe: bare `python` | negative control `dead` ×5 | passes |
| shadow chain: no git identity | timeline silently empty | passes |
| unpinned `anthropic` | every model call TypeErrors | passes |
| test verification: bare `pytest` | **15 tests fail as harness defects** | passes |
| `commit0` scorer: bare `python` | `fractional_score` → 0.0 | passes |
| git handles never closed | sweep dies after ~400 cells | passes (short runs) |

## Gate 0

Passes 5/5 on the experiment host. Failed 3/5 on first contact with a clean
machine, correctly, because of the bare-`python` defect — see the bug table.

## Pilots (development slice, exploratory, excluded from any reported frame)

| pilot | n | spend | observations | episodes (declared unit) | λ̂ | bearing |
|---|---|---|---|---|---|---|
| 10-instance | 10 | $11.48 | 62 | 3 | 0.30 | 10% |
| 40-instance (quarter of oracle dead) | 40 | $48.11 | 206 | 9 | 0.23 | 5% |
| 40-instance corrected | 40 | $50.96 | 204 | 0 | 0.00 | 0% |
| **pooled** | **80** | **$99.07** | **410** | **9** | **0.113** | **2.5%** |

Pooled bearing 95% CI (Clopper–Pearson): **[0.3%, 8.7%]**. Distinct onset
trees: **2**. Pre-registered gate (λ̂ ≥ 0.30 AND bearing ≥ 25%): **FAILS**.

Instrument-verification control: re-scoring pilot40's archived trees with the
corrected instrument reproduces 8 and 6 episodes exactly, same hole counts. So
the second pilot's zero is a property of the runs, not of the instrument.

## Test suite

547 tests, lint clean, **on the experiment host** — not only on the machine
they were written on. Before the bare-`pytest` fix, 15 of them failed there.

## Observation grid

| grid | observations | adjacent pairs | source |
|---|---|---|---|
| per-attempt, 5 local runs | 10 | 5 | measured |
| per-tool, same shape | 22 | 17 | measured |
| per-tool, 40 instances | 206 | 166 | pilot |

## The one-environment seam (built 2026-08-26)

| fact | value | source |
|---|---|---|
| test suite | 592 passing, lint clean, on the experiment host | box run |
| catalogue | 26 defects (20 found by use + 6 by the 20-agent audit) | review doc |
| golden check, psf__requests-5414 | gold → resolved=True (F2P 1/1, P2P 130/130); null → False | goldencheck run |
| golden check runtime | 34 s, $0 model spend | same |
| grade network stance | grade containers bridge (official conditions); measurement containers severed | code + first golden run (126/130 under `none`) |
| official pass semantics | PASSED\|XFAIL pass; SKIPPED fails | fetched upstream grading.py |
| defects caught by the gate itself, pre-spend | 3 (router workdir default; stale-ledger skip; grade network isolation) | goldencheck sessions |

## Pilot 40c — the first valid measurement (2026-08-26, exploratory, dev slice)

| fact | value | source |
|---|---|---|
| cells | 40, all routed + graded | ledger |
| spend | $34.24 | ledger |
| resolve rate | 13/40 = 32.5%, CI95 [18.6%, 49.1%] | official grader per cell |
| completed (Monitor) | 22/40; step progress 25/40 | ledger |
| observations / replays | 227, replays == observations on every cell | evidence |
| episodes (declared unit) | 0; bearing 0/40, CI95 [0%, 8.8%] | evidence |
| gates (λ̂ ≥ 0.30, bearing ≥ 25%) | both FAIL → pre-declared SWITCH to SWE-bench-Live | prereg rule |
| rescore control | canary bearing run reproduces 8 raw / 3 declared exactly under this instrument | rescore run |
| oracle holes | 21 / ~3,300 graded members, 3 instances, all explained | evidence |
| rollback exercise | 22/40 runs rolled back at least once | ledger |
| decoupling | flask-5014 resolved=True while Monitor scored the run failed | ledger+evidence |
