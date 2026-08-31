# What the Final Patch Hides: Event-Level Regression Measurement for Coding-Agent Harnesses

*Submission draft v4. The converter drops this line. Arms are named (rollback, repair-in-place, no recovery).*

---

## Abstract

When a coding agent is graded, only its final patch is examined. Does the agent break the repository's existing tests while it works, and does the final patch show it? We built an instrument that commits the working tree after every mutating tool call and replays the repository's previously-passing tests at every commit inside the benchmark's pinned container, and applied it to an unmodified public scaffold: mini-swe-agent 2.4.6, keeping its own loop, prompts, model layer, budgets, and submission protocol, with only its environment object replaced. On 40 SWE-bench Verified tasks it resolves 34 of 40 with `claude-sonnet-4-6` and 33 of 40 with `gpt-5.6-sol`, ends all 80 runs with a changed tree, and breaks previously-passing test functions in 11 of them — 135 counted per run, 100 distinct — of which exactly one is still failing when the official grader reads the final tree. Its own testing is not what hides them: over the window in which each test was broken, no test command ran at all in 6 of the 11 breaking edits, and in 4 a run executed the broken test and printed its failure by name. The undercount holds in our own harness under rollback (146 test functions broken across 8 of 47 runs, one still failing) and on a 40-task SWE-bench Live slice, with two caveats we make measurable: rollback leaves clean trees partly because 17 of its 37 runs ship a tree they never changed, and a model-stack difference we measured on Verified does not survive a change of scaffold. The blindness runs both ways — nine of the ten trees that killed the suite outright recorded no timeline event. Gating each step on the repository's own tests instead resolves 65% with no contaminated final tree, and 70% when the gate reads only half of those tests, with no failure among the 2,585 previously-passing ids it never read. Appendix A catalogues 28 measurement defects met while building the instrument, 23 of which produced plausible numbers rather than errors.

---

## 1. Introduction

Benchmarks for coding agents grade the final patch: SWE-bench [1] and its Verified subset [2] apply the submitted patch and report whether the target tests pass and the previously passing tests still pass; work on agent-caused regressions counts the previously-passing tests the submitted patch breaks [8]. That convention cannot see what happens while the agent works — an agent that breaks a test at step three and repairs it at step five submits a clean patch — so two questions have no answer: how often do agents break existing behaviour during a run, and how much survives to the final patch?

We answer both by measuring the timeline instead of the endpoint. Every mutating tool call becomes a commit on a git reference the agent cannot see; after the run, the repository's previously-passing tests are re-run at every commit inside the benchmark's pinned container. This is the observation primitive an agent operating system needs before it can build a recovery primitive. Our contributions are:

1. **An instrument for event-level regression measurement**, applicable to an agent we did not write: every mutating tool call becomes a commit on a hidden git reference, and the instance's previously-passing tests are replayed at every recorded state inside the benchmark's pinned container. Validity rests on controls, a liveness check, a golden check, and a re-scoring control (Section 3).
2. **The measurement on an unmodified public scaffold.** mini-swe-agent 2.4.6 keeps its own loop, prompts, model layer, budgets, and submission protocol; only its environment object is ours. It resolves 34 and 33 of the same 40 instances under two model families, ends all 80 runs with a changed tree, and breaks 135 previously-passing test functions counted per run (100 distinct) across 11 of them, one of which is still failing at grade time. In 6 of the 11 breaking edits its own test commands never ran the test it broke (Section 5.1).
3. **The same measurement across recovery policies, and what the clean tree costs.** The undercount holds in our harness and on SWE-bench Live (Sections 5.2, 5.6), but nearly half of the rollback arm's runs ship a tree they never changed; a harness-enforced regression gate resolves 65% with no contaminated tree, and 70% under a gate reading half the previously-passing ids, leaving no failure among the half it never read (Sections 5.4, 5.5).
4. **A catalogue of 28 measurement defects** found while building the instrument, classified by mechanism (Appendix A); 23 produced a plausible number rather than an error.

All measurements are exploratory, on development slices excluded from future confirmatory study.

## 2. Background and related work

**Benchmarks and their validity.** SWE-bench [1] grades a patch by tests that must go from failing to passing and tests that must keep passing; Verified [2] is the human-screened subset, deprecated by its maintainers as a capability metric [46]. Concerns about it have accumulated — contamination and memorisation [6], plausible-but-wrong patches [37], insufficient tests [7] — with rolling [3, 5], private [4], and long-horizon [23] alternatives in response. TDAD [8] measured regressions in *submitted* patches on Verified — 6.1% of previously-passing tests, 30.2% of non-empty patches — and reduced them with an impact-analysis map. Process-level benchmarks score intermediate states from logs [35, 36], AgentLens [38] naming regression cycles in 10.7% of passing OpenHands runs. A log-derived rate is bounded below by what the agent chose to run; we execute the repository's tests at every mutating call instead.

**Agent scaffolds and recovery.** SWE-agent [9], OpenHands [10], Agentless [11], and mini-swe-agent [12] span the scaffold design space [13, 14]. Within-episode recovery has been studied as self-reflection [15, 16], progress-gated recovery [17], checkpoint repair [18], and aligned checkpoints [41], with provenance-based recovery surveyed in [19]. Closest to us, Kim et al. [39] match logged intermediate edits against the gold diff and recover agents that reach a gold-identical patch and then destroy it; Gao et al. [40] bind verifier evidence to exact code states; neither counts regressions or compares recovery policies. Running the repository's tests during a run is deployed practice [43, 44]. Operating-system framings [20] and the branch-context primitive [42] supply scheduling and fork, commit, and abort, but do not observe the tree between actions. The regression problem is program repair's plausible-versus-correct patch problem [26, 28, 29], where regression tests are the main defence against overfitting [27]; trajectory-level evaluation [21, 22] argues resolve rate alone is uninformative. Public leaderboard trajectories record actions but not the tree at each step; the instrument commits the tree.

## 3. Instrument

![The harness and the instrument. Every mutating tool call commits the tree to a hidden git reference; after the run, the instance's previously-passing tests are replayed at every state. The test row shows one regressing at s₃–s₄ and repaired by a rollback to s₂, which the official grader, reading only sₙ, never sees. In the gated arm (dashed), the timeline's tests serve as the monitor.](fig_system.pdf){width=0.62}

Figure 1 shows the harness and the instrument. **Observational timeline.** After every mutating tool call the working tree is committed to a git reference outside the agent's view (a private index keeps the agent's own `git status` unchanged); rollbacks and the end of the run are observations too, and an archived run can be re-measured without re-running the agent. The agent's tools and the harness's checks execute inside the same pinned container as the replay, with file changes synchronised to the host tree the timeline records (Appendix A.2).

**Exhaustive replay.** For each observation the instance's previously-passing (PASS_TO_PASS) tests run inside the pinned container against that observation's tree; a regression event is a test that passes at one observation and fails at a later one. Every observation is replayed rather than bisected, because a recovery policy makes verdicts non-monotone. Replay is scoped to the files holding those tests, which keeps it affordable on Verified (about 26 CPU-minutes per sweep); Live's larger oracles cost tens of minutes per instance. Infrastructure failures during replay, and tests already failing in the unmodified image, are recorded as missing observations rather than failures.

**Validation.** Five checks run before any paid experiment (negative and positive controls, a flake screen, an unknown-rate ceiling, a baseline liveness check), plus a golden check that drives the gold patch through the agent's real tool path and a re-scoring control that re-measures an archived timeline with no model calls; re-scoring four arms' bearing runs reproduced their episode counts and probe sets exactly, and supplies the matched-unit columns of Table 1. Appendix C gives the detail.

## 4. Experimental setup

**Benchmark and slice.** SWE-bench Verified is a measurement substrate here, not a leaderboard: its pinned images and previously-passing test sets make the measurement possible. A 40-instance development slice (16 from django), stratified and fixed before any measurement, was used throughout and is excluded from future confirmatory study; the GPT-5.6 stack also ran its first 10 as a calibration, and plain rollback ran twice. SWE-bench Live [3], built monthly from live repositories, is the second substrate, graded by its own harness whose rules we mirror; its slice was fixed before any outcome (Appendix B) and its oracles are about twenty times larger (median 1,189 against 58 tests).

**Harness.** A planner decomposes the task into steps with verification commands; a worker executes each with three tools (read, write, shell); a monitor runs the verification, and on failure the recovery policy acts: **rollback** (reset to the last verified checkpoint and retry), **repair-in-place** (retry from the failed tree), or **no recovery** (the monitor's FAIL is final and the run ends). Runs have a $4 work-cost cap. One instance under one policy is a cell; each (model stack, policy, substrate) triple is an arm.

**Public scaffold.** mini-swe-agent 2.4.6 [12] runs unmodified — its own loop, prompts, model layer, 250-step and $3 billed-cost caps, and submission protocol. Only its environment object is replaced, so every bash command executes in the benchmark's pinned container and the tree is committed to the hidden reference after each command. The seam is inert rather than invisible: the agent's `git diff` and `git status` show its own uncommitted edits and none of the instrument's commits, but the baseline the instrument creates is named, and 54 of the 80 runs saw that name in some output (Appendix C). Six departures from the scaffold's leaderboard configuration are listed there too; we claim no equivalence to its published resolve rate, and because its budget differs in kind from our cap, no comparison between the two harnesses is tested. A grid census (Appendix C) puts its sampling inside our arms' range — a median of 3 and 2 tree-changing observations per run against our 2 to 5 — while it issues 9 to 15 bash commands per observation, most of them reads.

**Models.** Two frontier stacks under identical harness, policy, instances, and caps: `claude-opus-4-7` with `claude-sonnet-4-6`, and `gpt-5.6-sol` with `gpt-5.6-terra` (planner, worker), recorded in every run manifest.

**Outcomes.** Resolve is the official grader's verdict on the final patch. Events are reported at three levels because a few runs produce most: incidents (observations at which at least one test broke, the co-primary endpoint), declared events (one per test function and onset, parametrised variants collapsed), and bearing runs (runs with at least one event). Because an event and a final-state failure are different objects, the undercount is stated in matched units: test functions broken during a run against those still failing at the end. Final-state contamination counts previously-passing tests failing in the graded patch, net of baseline-dead tests; a tree that kills the suite is graded as failing every test but is a missing observation for event counting, so a contaminated cell need not be a bearing run.

**Pre-declaration.** The event unit, the proceed criteria, and the recovery-policy contrast were declared before the relevant data existed, and the analysis was committed before either comparison arm finished; one amendment preceded unblinding and one grading correction followed it (Section 5.4, Appendix B).

## 5. Results

### 5.1 An unmodified public scaffold breaks tests and puts them back

mini-swe-agent 2.4.6 ran unmodified on the same 40 instances. With `claude-sonnet-4-6` it resolved 34 of 40 (35 rot-aware) over 2,101 bash commands for $31.38, four runs ending at its own $3 cap — three of those still grade resolved, because resolve is read off the final tree, not off the patch the scaffold never submitted (Appendix C); with `gpt-5.6-sol`, 33 of 40 (34 rot-aware) over 937 commands for $12.13, every run ending by its submission protocol. No run of the 80 ended with a tree byte-identical to its start; the four capped runs stopped before the scaffold's submission step, but all four had already changed the tree.

The timeline recorded 56 events in 5 of the 40 Claude runs and 79 in 6 of the 40 GPT-5.6 runs, on 8 instances, 3 bearing under both models. In matched units those 11 runs broke 135 previously-passing test functions counted per run — 100 distinct, since the two stacks break overlapping sets on the three instances bearing under both — and when the grader read the final tree exactly one was still failing: `test_pkgfile` in pytest-6197 (Claude 0 of 56; GPT-5.6 1 of 79). The scaffold has no monitor, no checkpoint, and no rollback, so every one of those closures is an ordinary forward edit the agent chose to make. They are short and concentrated: the eleven stayed open for 1 to 22 bash commands (median 4), and three runs supply 81% of the 135. Most of what the endpoint misses here is a transient inside a single logical edit, not damage the agent lived with.

**Would its own testing have caught it?** For each breaking edit we read the scaffold's trajectory over the window in which the test was broken — the breaking command up to, but not including, the repair — and ask whether a test runner ran, whether any run would have executed *this* test, and whether its output showed the failure (Appendix C; coverage is three-valued, *unknown* for runs that stopped early or were truncated). Of the 11 breaking edits, 6 saw no test command run at all while the regression was open; in 4 a run executed the broken test and printed its failure by name; 1 is unknown. What the endpoint misses here is not red output the agent ignored: it is regressions the agent's own test commands never ran.

### 5.2 The same undercount in our harness, and what the clean tree costs

![Every run that broke a previously-passing test, in matched units: test functions broken on the timeline against those still failing when the grader reads the final tree (dark). Left, the unmodified public scaffold under both models; right, our rollback arm, where hatched bars are runs whose final tree is byte-identical to their start.](fig_undercount.pdf){width=0.62}

Under rollback the Claude stack resolved 13 of 40 (32.5%, exact 95% CI 18.6% to 49.1%) for $34.24 and the GPT-5.6 stack 13 of 40 for $8.14; a circuit breaker stopped three of its cells and a synchronisation defect lost two, so 37 were attempted and 35 graded, and one instance is unresolvable under the current official parser (Appendix A.3). The median rollback run makes 4 mutating tool calls under GPT-5.6 and 3 under Claude.

Eight of the 47 GPT-5.6 rollback runs (the sweep and a 10-instance calibration) broke a previously-passing test: 146 test functions counting each run separately (145 distinct), of which one — `test_can_read_toml_env_variable` in pylint-6386 — was still failing at grade time. Three storms supply 88% of the 184 declared events, reported as exposure only. Figure 2 shows every bearing run on both sides.

Not every clean tree is work. Excluding the hidden reference, 17 of the 37 rollback runs end byte-identical to their start, as do 19 of the Claude stack's 40, 4 of 40 under the gate, and 0 of 40 under repair-in-place. Four of the six bearing sweep runs shipped nothing; restricted to the two that delivered a patch, 1 of 55 broken test functions is still failing — which is why the public-scaffold arm, none of whose 80 runs ends unchanged, carries the claim.

The blindness runs both ways. A break that stops the suite from being collected opens no timeline event, because a test the parser never mentions is a missing observation rather than a failure, while official grading scores such a patch as failing every test. Nine of the ten suite-killed trees across all arms recorded no timeline event at all: the two instruments have complementary blind spots.

### 5.3 On Verified, a model-stack difference that does not generalise

On the same 40 instances, under the same harness and rollback policy, the Claude stack produced no regression events in 40 runs, the GPT-5.6 stack 140 in 6 of 37 (one-sided Fisher p = 0.010). It does not generalise: the same `claude-sonnet-4-6` worker under unmodified mini-swe-agent is bearing on 5 of 40 instances with 56 events (Section 5.1), and the difference is not detected on Live. We report it as a measurement on one slice, not a property of either model; model and adapter are confounded, and three controls bounding it are in Appendix C.

### 5.4 Recovery policy: what the clean final tree costs

On the primary endpoint, final-state contamination paired by instance, no recovery was worse than rollback on 9 instances and better on 1 (exact sign test p = 0.021); repair-in-place was worse on 5, better on 1 (p = 0.22). Incident exposure, the co-primary, did not differ (p = 0.45 and 1.0). The arms cannot show that the policy *caused* the difference: no recovery ends the run at the monitor's first FAIL, so its regressions arrive on the last tree-changing observation with nothing after them, and its 19-of-19 persistence is the stopping rule as much as the policy.

Rollback pays for its clean tree in delivered work: it resolved 13 of 40 against 24 and 22, losing nine discordant pairs to repair-in-place and winning one (McNemar p = 0.021). The loss is the monitor's — of the 18 rollback runs that failed, 15 failed at the first step, and 9 of those instances were resolved by an arm that kept work the check had rejected and the grader accepted — and 17 of 37 runs ended with a tree they never changed.

**Disclosure.** The first unblinding gave p = 0.375, because seven cells whose final trees killed the suite had been dropped as ungradable; correcting the grader and re-grading those archived trees gives the 0.021 above — a rule change made after the data were seen (Appendix B).

### 5.5 On Verified, regression-gated rollback removes the tradeoff

A fourth arm, added after unblinding (exploratory), replaces the planner-written check with the repository's previously-passing tests, run after every attempt: a step is rejected only if a test that passed at the start now fails. It resolved **26 of 40 (65.0%)**, above rollback's 13 and the ungated arms' 24 and 22, below the 34 and 33 unmodified mini-swe-agent reaches under a budget differing in kind (Table 1), with **no contaminated final tree** and 4 empty patches against rollback's 17. Against rollback on 35 shared instances it won 10 discordant pairs and lost 1 (p = 0.012); against repair-in-place it resolves 26 to 24 and is better on contamination 5 to 0. What the gate offers is a guarantee rather than a rate, and it is partly circular: its tests are the benchmark's own previously-passing set (median ratio 1.00), so a clean tree follows whenever the gate saw the final tree and the grader reads what the gate read. A fifth arm weakens the guarantee: the gate reads a deterministic half of the ids (2,278) and never the other half (2,585), on which the grader also rules [45]. The split is over ids, not files, so a break reaching a held-out id usually reaches a watched one too: the held-out half bounds the leak rather than sampling it independently. It resolved **28 of 40 (70.0%)**, indistinguishable from the full gate (p = 0.73) and better than rollback (13 to 2, p = 0.007), with no held-out failure apart from the parser-capped instance.

### 5.6 Replication on SWE-bench Live

Table: Every arm on both substrates. **empty**: runs whose final tree is byte-identical to their start. **events**: declared (test function, onset) pairs — exposure only, never a denominator. **broken → left**: over the bearing runs, previously-passing test functions broken during the run against those still failing at grade time. The substrates score an absent id as their harnesses do: on Verified a graded test the log never mentions counts as failing, on Live it does not. **contam.**: graded cells whose final tree fails a previously-passing test, net of baseline-dead tests (those already failing in the unmodified image); no recovery's 9 is six suite-killed trees and three partial, and the Claude arm's 1 is baseline rot rather than agent damage (Appendix C). The public-scaffold rows ran under mini-swe-agent's own budget, not our cap, so no comparison between harnesses is tested. Definitions, incidents, and per-row provenance: Appendix C.

| substrate | arm | resolved | empty | events | bearing | broken → left | contam. | spend |
|---|---|---|---|---|---|---|---|---|
| Verified | mini-swe-agent, Claude | 34 | 0/40 | 56 | 5/40 | 56 → 0 | 0 | $31.38 |
| Verified | mini-swe-agent, GPT-5.6 | 33 | 0/40 | 79 | 6/40 | 79 → 1 | 1 | $12.13 |
| Verified | GPT-5.6 rollback | 13 | 17/37 | 140 | 6/37 | 104 → 1 | 1 | $8.14 |
| Verified | Claude rollback | 13 | 19/40 | 0 | 0/40 | — | 1 | $34.24 |
| Verified | GPT-5.6 gated | 26 | 4/40 | 70 | 5/40 | 62 → 0 | 0 | $10.51 |
| Verified | GPT-5.6 split-gated | 28 | 3/40 | 14 | 3/40 | 14 → 0 | 0 | $9.40 |
| Verified | GPT-5.6 repair-in-place | 24 | 0/40 | 84 | 7/40 | 83 → 4 | 5 | $14.73 |
| Verified | GPT-5.6 no recovery | 22 | 1/40 | 19 | 3/40 | 19 → 19 | 9 | $4.40 |
| Live | GPT-5.6 rollback | 0 (2) | 25/39 | 156 | 6/39 | 147 → 0 | 0 | $10.29 |
| Live | Claude rollback | 2 (3) | 24/39 | 49 | 3/39 | 48 → 0 | 1 | $49.80 |
| Live | GPT-5.6 gated | 1 (4) | 9/39 | 280 | 3/39 | 270 → 0 | 2 | $11.49 |
| Live | GPT-5.6 no recovery | 4 (6) | 3/39 | 30 | 4/39 | 30 → 30 | 4 | $8.95 |

Live keeps the undercount and loses both the resolve rates and the stack difference (Table 1). Resolve collapsed for every arm, to at most 6 of 40 even rot-aware against the same four arms' 13 to 26 on Verified, so the recovery contrast on resolve is uninformative here; the collapse is consistent with contamination of Verified [6, 46], though the substrates also differ in oracle size and rot, and 25 of 39 rollback runs shipped an unchanged tree. The undercount replicated: rollback broke 147 test functions across 6 of 39 runs and left none failing, while no recovery left every one of its 30 (sign p = 0.125). The gate, clean by construction on Verified, left two contaminated trees here (Appendix C), and the stack difference did not replicate (3 of 39 against 6, p = 0.24).

## 6. Discussion

**Why timeline regressions matter.** A repaired regression is not free — 27% of what the eight Verified bearing runs cost went to work done while one was open — and an endpoint-only benchmark cannot tell a run that broke nothing from one that broke 49 tests and repaired them. **Limitations.** Every rate is an estimate on a 40-instance slice with one sweep per arm, and event counts are lower bounds in both directions (Appendix C).

---

### References

[1] C. Jimenez et al. SWE-bench: Can language models resolve real-world GitHub issues? ICLR 2024. arXiv:2310.06770.
[2] OpenAI. Introducing SWE-bench Verified. OpenAI blog, August 2024. openai.com/index/introducing-swe-bench-verified.
[3] L. Zhang et al. SWE-bench Goes Live! NeurIPS 2025 Datasets and Benchmarks. arXiv:2505.23419.
[4] X. Deng et al. SWE-Bench Pro: Can AI agents solve long-horizon software engineering tasks? arXiv:2509.16941, 2025.
[5] I. Badertdinov et al. SWE-rebench: An automated pipeline for task collection and decontaminated evaluation of software engineering agents. NeurIPS 2025. arXiv:2505.20411.
[6] S. Liang et al. The SWE-Bench Illusion: When state-of-the-art LLMs remember instead of reason. arXiv:2506.12286, 2025.
[7] B. Yu et al. UTBoost: Rigorous evaluation of coding agents on SWE-Bench. ACL 2025. arXiv:2506.09289.
[8] P. Alonso, S. Yovine, and V. A. Braberman. TDAD: Test-driven agentic development: Reducing code regressions in AI coding agents via graph-based impact analysis. arXiv:2603.17973, 2026.
[9] J. Yang et al. SWE-agent: Agent-computer interfaces enable automated software engineering. NeurIPS 2024. arXiv:2405.15793.
[10] X. Wang et al. OpenHands: An open platform for AI software developers as generalist agents. ICLR 2025. arXiv:2407.16741.
[11] C. S. Xia et al. Agentless: Demystifying LLM-based software engineering agents. FSE 2025. arXiv:2407.01489.
[12] K. Lieret and C. E. Jimenez. mini-swe-agent. Software, 2025. github.com/SWE-agent/mini-swe-agent.
[13] X. Wang et al. Executable code actions elicit better LLM agents. ICML 2024. arXiv:2402.01030.
[14] S. Yao et al. ReAct: Synergizing reasoning and acting in language models. ICLR 2023. arXiv:2210.03629.
[15] N. Shinn et al. Reflexion: Language agents with verbal reinforcement learning. NeurIPS 2023. arXiv:2303.11366.
[16] A. Madaan et al. Self-Refine: Iterative refinement with self-feedback. NeurIPS 2023. arXiv:2303.17651.
[17] A. Kadu and A. Krishnan. ReflexGrad: Within-episode failure recovery in LLM agents via progress-gated dual-process routing. arXiv:2511.14584, 2025.
[18] P. Mazaheri. REPOT: Recoverable program-of-thought via checkpoint repair. arXiv:2605.30052, 2026.
[19] Y. Wang et al. From agent traces to trust: A survey of evidence tracing and execution provenance in LLM agents. arXiv:2606.04990, 2026.
[20] K. Mei et al. AIOS: LLM agent operating system. COLM 2025. arXiv:2403.16971.
[21] S. Kapoor et al. Holistic Agent Leaderboard: The missing infrastructure for AI agent evaluation. arXiv:2510.11977, 2025.
[22] R. Shu et al. What resolve rate hides: Trajectory structure diagnostics for coding agents. arXiv:2607.06184, 2026.
[23] T. Le et al. SWE-EVO: Benchmarking coding agents in long-horizon software evolution scenarios. arXiv:2512.18470, 2025.
[24] M. Raghavendra et al. Agentic rubrics as contextual verifiers for SWE agents. ACL 2026. arXiv:2601.04171.
[25] M. L. Dihan and M. A. R. Khan. SWE-Shepherd: Advancing PRMs for reinforcing code agents. arXiv:2604.10493, 2026.
[26] Z. Qi, F. Long, S. Achour, and M. Rinard. An analysis of patch plausibility and correctness for generate-and-validate patch generation systems. ISSTA 2015. doi:10.1145/2771783.2771791.
[27] Y. Lou et al. When automated program repair meets regression testing: An extensive study on two million patches. ACM TOSEM 33(7), 2024. arXiv:2105.07311.
[28] Z. Fei et al. Patch correctness assessment: A survey. ACM TOSEM 34(2), 2025. doi:10.1145/3702972.
[29] H. Ye, M. Martinez, and M. Monperrus. Automated patch assessment for program repair at scale. Empirical Software Engineering 26(2), 2021. doi:10.1007/s10664-020-09920-w.
[30] SWE-bench issue #601: Test result hijacking via stdout forging in evaluation harness. github.com/SWE-bench/SWE-bench/issues/601, June 2026.
[31] OpenHands issues #4235 (October 2024) and #7044 (March 2025): Docker and environment errors in SWE-bench instance evaluation. github.com/OpenHands/OpenHands.
[32] J. Yang et al. SWE-smith: Scaling data for software engineering agents. NeurIPS 2025 Datasets and Benchmarks. arXiv:2504.21798.
[33] J. Pan et al. Training software engineering agents and verifiers with SWE-Gym. ICML 2025. arXiv:2412.21139.
[34] S. Liu et al. Context as a tool: Context management for long-horizon SWE-agents. Findings of ACL 2026. arXiv:2512.22087.
[35] M. B. Madiraju and M. S. P. Madiraju. RigorBench: Benchmarking engineering process discipline in autonomous AI coding agents. arXiv:2606.22678, 2026.
[36] J. Chen et al. SWE-CI: Evaluating agent capabilities in maintaining codebases via continuous integration. arXiv:2603.03823, 2026.
[37] Y. Wang, M. Pradel, and Z. Liu. Are "solved issues" in SWE-bench really solved correctly? An empirical study. ICSE 2026. arXiv:2503.15223.
[38] P. Sahoo et al. AgentLens: Revealing the lucky pass problem in SWE-agent evaluation. arXiv:2605.12925, 2026.
[39] M. Kim et al. Coherence collapse: Diagnosing why code agents fail after reaching the right code. arXiv:2603.24631, 2026.
[40] X. Gao, J. Yang, and Q. Yang. Looping is not reliability: State-bound evidence and typed revision contracts for agentic code repair. arXiv:2607.24604, 2026.
[41] Y. Zhuang et al. AgentRewind: Recoverable execution for long-horizon LLM agents. arXiv:2608.14380, 2026.
[42] C. Wang and Y. Zheng. Fork, explore, commit: OS primitives for agentic exploration. arXiv:2602.08199, 2026.
[43] Y. Chen, T. Ahmed, R. Jabbarvand, and M. Hirzel. Can old tests do new tricks for resolving SWE issues? FSE 2026. arXiv:2510.18270.
[44] P. Gao et al. Trae Agent: An LLM-based agent for software engineering with test-time scaling. arXiv:2507.23370, 2025.
[45] T. Ahmed, J. Ganhotra, A. Shinnar, and M. Hirzel. Investigating test overfitting on SWE-bench. arXiv:2511.16858, 2025.
[46] OpenAI. Why SWE-bench Verified no longer measures frontier coding capabilities. OpenAI blog, February 2026. openai.com/index/why-we-no-longer-evaluate-swe-bench-verified.

## Appendix A: The measurement defect catalogue

### A.1 The catalogue by mechanism

Each row is a defect found while building the instrument. **S**: it produced a plausible number rather than an error. **G**: it was present while the test suite passed (A4 is the suite). **H**: it was findable only on a clean host. Class A rows depend on the machine the code runs on; class B rows render an infrastructure failure as a measurement; class C rows measure something other than the event as defined; class D rows consume a producer's output without checking the producer.

| # | Defect | Consequence | S | G | H |
|---|---|---|---|---|---|
| A1 | Shadow commits inherited the machine's git identity | timeline silently empty on any clean machine | ✓ | ✓ | ✓ |
| A2 | Validation probe invoked bare `python` | every probe exit 127 on a clean host | ✗ | ✓ | ✓ |
| A3 | Unpinned SDK resolved a different major version | two machines ran different code | ✗ | ✓ | ✓ |
| A4 | Test fixtures verified with bare `pytest` from PATH | 15 tests fail on a clean host as harness defects | ✗ | — | ✓ |
| A5 | A benchmark scorer invoked bare `python` | score 0.0 from a missing interpreter | ✓ | ✓ | ✓ |
| A6 | Agent executed on the host; measurement in the container | 26 of 28 zero-step runs; see A.2 | ✓ | ✓ | ✓ |
| B1 | Output marker used shell `:` | a 13/13 run scored as 13 errors | ✓ | ✓ | |
| B2 | Results on stderr, markers on stdout | one framework's results outside the parsed slice | ✓ | ✓ | |
| B3 | Probe diffed against a deleted commit | every graded test a hole | ✓ | ✓ | |
| B4 | Interrupted mirror clone accepted with zero commits | later instances fail far from the cause | ✓ | ✓ | |
| B5 | Container workdir unvalidated | every exec exit 127 | ✓ | ✓ | |
| B6 | Isolation removed loopback, not only the internet | 23.5% of the oracle dead | ✓ | ✓ | |
| B7 | Provider cache served removed containers | later cells replay against nothing and score clean | ✓ | ✓ | |
| B8 | Tree reset reverted image build-time edits | one repository family's oracle dead | ✓ | ✓ | |
| C1 | Bisection assumed monotone verdicts | recovered regressions invisible | ✓ | ✓ | |
| C2 | Declared event unit not implemented | rate off by up to 2.7× | ✓ | ✓ | |
| C3 | Rollbacks produced no observation | recoveries invisible to the timeline | ✓ | ✓ | |
| C4 | "Persists past the step boundary" undefined | one reading excludes the events of interest | ✓ | ✓ | |
| D1 | Audit field never populated | 0 tool errors reported, always | ✓ | ✓ | |
| D2 | Malformed planner output crashed the run | 33% of runs lost as task failures | ✗ | ✓ | |
| D3 | Dependency install counted as agent work | attribution vacuous | ✓ | ✓ | |
| D4 | Join took the first observation, not the last | failures pinned to the wrong tree | ✓ | ✓ | |
| D5 | Executed config rebuilt from a name | manifest described a different run | ✓ | ✓ | |
| D6 | Git handles never released | sweeps die after ~400 cells | ✓ | ✓ | |
| D7 | Console re-walked the tree per run | presented as a dead server | ✗ | ✓ | |
| D8 | Detection events lacked the ids attribution joined on | attributed detection always unknown | ✓ | ✓ | |
| D9 | Absent coverage rendered as measured silence | unmeasured regressions reported as silent | ✓ | ✓ | |
| D10 | Output-capped turns executed | half-written file; a 78-event storm | ✓ | ✓ | |

Totals: 23 of 28 silent; 27 of 28 present under a passing suite; 6 of 28 findable only on a clean host. Ten further defects found after this census closed are described where they arose (Sections 5.2 and 5.4, Appendices A.3 and C); the rest will be released with the code.

### A.2 The defect that invalidated a conclusion

After nineteen fixes, every validation check passed, the re-scoring control reproduced archived episode counts exactly, and three pilots (80 runs) measured an event rate far below the pre-declared proceed criterion. We drafted the conclusion the rule directed: the benchmark offered too little opportunity, so switch benchmarks. The conclusion was wrong. The harness ran the agent on the host in a bare source checkout, while every probe and every gate ran inside the pinned container. On the host, `import matplotlib` in that checkout succeeds by importing the uncompiled source tree as a namespace package, and everything downstream fails in ways indistinguishable from an incompetent agent. Twenty-six of 28 zero-step runs traced to this. The checks had validated the measurement path and never the agent's execution path. The fix was to route the agent's tools and checks through the same container as the replay, and to add a per-cell environment parity check that runs before any model call.

### A.3 Mechanisms in public infrastructure

Class A: OpenHands issues #4235 and #7044 [31], environment construction failures against SWE-bench images. Class B: UTBoost's finding that the held-out oracle is insufficient at leaderboard scale [7], and, on SWE-bench Live, baseline tests that fail in the unmodified image because the calendar passed a deprecation date encoded in the instance. Class C: final-state regression measurement [8]. Class D: the SWE-bench grader accepting forged test output on stdout [30], and, at the harness's current revision, a parser that keys parametrised ids by their full text while the dataset stores them truncated at the first space (the earlier parser's behaviour), so 16 previously-passing ids of one instance in our slice match no output and every submission to it grades as unresolved.

## Appendix B: Pre-declaration and analysis

The event unit is one (test function, onset observation) pair with parametrised variants collapsed. The gates for proceeding on a substrate were an event rate of at least 0.30 per run and a bearing fraction of at least 25%. Both stacks failed the conjunction (bearing 0% and 16.2%); the gates were then re-declared per (substrate, model stack) before the contrast was unblinded, and no arm reported here is confirmatory. The contrast's primary endpoint is final-state contamination, paired by instance, exact two-sided sign test; the co-primary is incident exposure. The analysis script was committed before either comparison arm finished. The regression-gated arm was added after the contrast was unblinded and is exploratory. No multiplicity adjustment was planned or applied. Budget exhaustion is scored as failure. Instances whose baseline oracle records failures in the unmodified image are excluded from resolve-rate comparability claims, and their dead tests are typed as missing observations for event counting.

**Compute.** Every run, replay, and grade executed on one 12-vCPU, 22 GB x86 virtual machine running Docker, with no GPUs; the six harness arms on Verified in Table 1 cost $81.42 in model spend and the two public-scaffold arms $43.51, plus $6.87 for the second rollback sweep, the calibration and pilots a comparable amount, and the Live sweeps are reported with their own spend in Table 1.

**Slice construction on SWE-bench Live.** Fixed before any outcome: pytest-parsed instances with 27 to 3,000 previously-passing tests, at most four per repository, earliest first; the first 40 of 48 candidates whose image passed a model-free environment check. No coverage maps are built for Live, so attribution there is unknown by construction.

**Attribution (Verified only).** A detected regression is classified three ways: *attributed* if a failing harness check and the broken test both exercise a file the agent changed at that observation (per-instance coverage map at the base commit), *co-occurring* if some check failed while the regression was open without such a link, and *unknown* if the map cannot say. Of the calibration's 57 raw episodes, 39 were attributed to a failing check, 16 co-occurred with one, and 2 were silent; 48 of the sweep's 162 had none. The public-scaffold arm has no monitor, so attribution is undefined for it and no silence statistic is reported.

**Grading correction between unblindings.** The first unblinding of the recovery-policy contrast gave p = 0.375 for the primary endpoint: seven of the eight cells whose final trees made the suite uncollectable (five no-recovery, two repair-in-place) had been dropped as ungradable. Official grading scores such a patch as failing every test, so the most contaminated states had been dropped from the endpoint they bore on most. The grader, not the replay, now distinguishes an environment failure from a patch that kills the suite, and the seven archived trees were re-graded without re-running any agent.

## Appendix C: The public-scaffold arm

**The seam.** mini-swe-agent's loop has one place that touches the working tree: `DefaultAgent.execute_actions` calls `env.execute(action)`. We replace the environment object and nothing else, so the scaffold keeps its own loop, prompts, litellm model layer, step and cost limits, and submission protocol. Our environment runs each bash command inside the benchmark's pinned container, pulls the resulting tree back to the host the timeline records, and commits an observation when the tree changed. Two properties of the seam were established by measurement rather than assumed. The container's git state is not advanced by the instrument, so the agent's own `git status`, `git log`, and `git diff` show its uncommitted work and none of the instrument's commits; the first paid pilot, run before this was true, spent eight of twenty commands discovering that its edits had been committed underneath it. The instrument is not invisible, only inert: the synthetic baseline it creates is named `taste-baseline` and authored `taste`, and 54 of the 80 runs saw that name in some command's output, 8 of them through `git status` (departure 2). And the routed command is wrapped so that a command ending in a here-document terminator or a comment parses as the scaffold's own `bash -c` would parse it; before that fix, 39 of 40 GPT-5.6 cells hit a shell syntax error, because that model writes every script as a here-document.

**Departures from the scaffold's leaderboard configuration.** (1) The measurement container has no network, while its own deployment uses Docker's default bridge; the unchanged prompt tells the agent it may install tools, and here it cannot. (2) The container's git history is a single synthetic commit, because the image's upstream history can contain the fix the task asks for. (3) Resolve is graded on the final tree by the official grader rather than on the scaffold's submitted patch. (4) Our replay and grading run outside its loop. (5) The two models are reached through the scaffold's litellm layer, GPT-5.6 through its Responses-API model class, which the chat-completions class cannot serve (it returned three replies with no tool call and the scaffold exited on repeated format errors). (6) Prices come from our verified table rather than litellm's, which carried a stale entry for one model. For these reasons we claim no equivalence to its published resolve rate.

**Sampling grid.** Per run, median tree-changing observations and mutated files, with the run and final observations excluded because neither is an opportunity to see a transition: public scaffold 3 and 3 (Claude), 2 and 2 (GPT-5.6); no recovery 2 and 2; repair-in-place 4 and 4; rollback 5 and 7; gate 4 and 4. Observations at which more than one file changed are 0 to 6% in every arm. The scaffold issues 14.8 and 9.1 bash commands per observation against the harness's one tool call per observation, because most of its commands read rather than write.

**The self-test classifier.** For each episode we take the window in which the test was broken — the breaking command up to, but not including, the command that repaired it — and label three things over the scaffold's own trajectory. *Tested*: a test runner ran and its output proves it ran, so a runner that died on an import does not count. *Covered*: one of those runs would have executed this test, decided on the runner's positional arguments matched at a label boundary (a run of two sibling modules does not cover a test in a third), honouring `-k`, `-m`, `--deselect`, and explicit node ids. *Shown*: a covering run's output names this test on a line that also carries a failure marker. Coverage and display are three-valued: a run that stopped at an earlier failure under `-x`, or whose output was truncated by a pipe, is *unknown* rather than negative, because calling it negative would assert that the agent ran the test and was not told, from evidence that cannot support it. The classifier's labels were audited by hand against the raw trajectories of eight bearing cells from an earlier pilot of this arm (run before the two shell defects below were fixed, and therefore not among the runs reported here); the audit disagreed with an earlier version of the classifier on all three totals, and the version reported here reproduces the audit exactly.

**The gate's two contaminated Live trees.** One run hit the $4 cap before the gate saw its final tree; the other passed every check yet fails 13 grader tests, 7 in the file the hidden test patch rewrites.

**Uses.** The per-observation verdicts are the step-level ground truth that patch verifiers [24] and process reward models [25] require.

**Further limitations.** Event counts are lower bounds in both directions: a break that stops the suite from being collected opens no timeline event (Section 5.2), and no-recovery's 19-of-19 persistence is confounded with its stopping rule, since that arm ends at the monitor's first FAIL — a no-recovery arm that keeps executing after a failed check is the experiment that would separate policy from truncation. The public-scaffold arm is one scaffold at one version under two model families; its budget differs in kind from our arms' ($3 billed and 250 steps against a $4 work cost), so the two harnesses are placed side by side and never tested against each other. The self-test measurement rests on 11 breaking edits, one of them coded unknown. Cross-stack comparisons confound model with adapter. The gated arms are post hoc and select tests from benchmark metadata. Live resolve is near floor, and under rollback most Live runs ship nothing, so its recovery contrast is uninformative on resolve.

**Bounding the model-stack difference (Section 5.3).** Exposure differs about twofold, not 140-fold: in opportunity units the Claude arm makes 147 observations to the GPT arm's 218, with per-run medians of 3 observations and 3 mutated files against 5 and 7. The zero is measured, not a detection failure: re-scoring that stack's earlier canary run reproduces its three events. And a second plain-rollback sweep reproduced the GPT side (13 of 36 graded resolved, 116 events, 6 of 37 bearing), so the bearing fraction is stable even though per-instance event counts are not.

**The Claude rollback arm's contaminated tree.** django-14238, a run whose tree never changed. Grading its unmodified base tree three times returns 38 of 39 previously-passing tests each time: one graded id never appears in the log, which upstream's rule scores as a failure. It is baseline rot of the kind Appendix A.3 describes, not a regression the timeline missed.

**The two readings.** The graded prediction excludes test files, as the official harness restores them anyway, while the replay reads the tree as it stands, so an agent edit to a test file is visible to the timeline and not to the grader.

**Reading Table 1.** *Baseline-dead* tests already fail in the unmodified image; *rot-aware* resolve counts a run resolved when every target test passes and no failure lies outside that set. On Live, **resolved** is of 40 while **empty** and **bearing** are of the 39 attempted (torchtune-1806 exceeds the replay budget). **broken → left** omits any cell whose suite the grader could not collect, since such a cell would otherwise read as perfect capture; repair-in-place's row therefore covers six of its seven bearing runs and 83 of its 84 events.

**Incidents.** The pre-declared co-primary, per arm, in the order of Table 1: 5, 6, 11, 0, 7, 4, 7, 3, 9, 6, 6, 4.
