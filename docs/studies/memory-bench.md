# Memory layers compared under one setup

Registered on 2026-11-05, before any comparison run. The code is `studies/memory_bench/` and
`taste/memstore/`; the LongMemEval split is `data/splits/memory-bench-lme.json`. Settings still to be
chosen on the dev split are named as such below, and are frozen in a commit before the held-out part is
opened.

## Why

Scores for agent memory systems are mostly reported by each system's makers, each with its own reader,
judge, prompt and retrieval budget, and the same system's score moves by tens of points when those change.
A comparison means something only when everything but the memory is held fixed.

## Question

Under one reader, one judge and one token budget, how does a version-controlled memory with keyword search
and a small YAML header per commit compare with MemPalace and Mem0, against no memory and full context, on
public benchmarks, in retrieval and in final-answer accuracy?

## Systems

- **No memory**; **full context**, as much of the history as the reader's window holds, whatever the budget (31,000 tokens
  for Llama, whose window here is 32,768; 118,000 for gpt-oss, whose window is 131,072 and whose output cap on this row is 12,288, because on
  prompts over 100,000 tokens its reasoning used up a cap of 4,096 on 8 of 50 dev questions and 8,192 on 1, and left no answer): when it
  cuts, the newest sessions are kept, and how many were cut is reported per question; **the newest text**, the same list cut
  to the budget, as a context window would; **LongMemEval oracle** (the evidence sessions only).
- **Flat keyword**: the same units as Taste's store in a plain list, ranked by the same code
  (`taste/memstore/keyword.py`: letters and digits, Porter's stemmer, BM25 with k1 1.5 and b 0.75). Git stores identical
  bytes once, so the list searches the first copy of identical text only.
- **Flat dense**: the same units, the first copy of identical text only, ranked by cosine similarity under one shared local embedder,
  all-MiniLM-L6-v2.
- **Taste**: `taste.memstore`, a fresh store per question, one file per round, one commit per session, searched
  over contents with `Store.search_contents`; once with headers off, once with headers on. A header holds a
  one-sentence "what" and keywords written by the one auxiliary model from answer-free text, plus fields the
  commit knows itself.
- **MemPalace 3.11.0** (release commit `602d796c`): its own benchmark runner in raw mode, and its product path
  (mining and search), reported apart.
- **Mem0 open source 2.2.1**: a locally served model, a local embedder and a local vector store, with no
  route to any paid service. One `add` with inference per round, as Mem0's own LongMemEval harness does, with a system line
  giving the session's date and the same date through `add`'s `prompt` argument, which the release sends as custom instructions.
  Without that its extraction prompt says the messages were observed today, whatever the system line says: on LoCoMo's dev
  conversations, in the first run with each, the date took F1 from .453 to .492 and temporal questions from .280 to .518, with nothing patched, so it is this
  system's setting. `search` with `top_k` 1,000 and `threshold` 0, the lowest it allows, so memories it scores below zero do not come back: on LongMemEval's dev questions a
  search returned a median of 485 of the 694 memories written (on the 46 questions with a process of their own), no question falls under 90% of 8,000 tokens for it, and 7 of 50 fall under 90% of 16,000. Each memory goes to the reader as written, in Mem0's rank
  order until the budget is full. At first this asked for 200, which on LoCoMo's short memories ran out at about 6,000 tokens, so dev
  was run again. LoCoMo's `speaker_a` is the user and `speaker_b` the assistant, as in Mem0's harness. gpt-oss-120b reasons inside its
  output cap, so the cap is 4,096 tokens with reasoning effort low: at Mem0's default of 2,000 the reasoning used it up on 84 of 776
  extractions, and at 4,096 all 13,709 practice replies parsed. Mem0's makers recommend an embedder of at least Qwen 600M for hybrid
  search, so Qwen3-Embedding-0.6B was tried on LoCoMo's two dev conversations beside all-MiniLM-L6-v2, the embedder every system here
  shares. Session recall at five came out .863 against .865 and F1 with gpt-oss-120b reading .488 against .511, so the held-out row keeps
  all-MiniLM-L6-v2, and the Qwen row is not run on held-out.

One auxiliary model serves every system that needs one (Mem0's extraction, Taste's headers): gpt-oss-120b if
its JSON replies parse at least 95% of the time on 50 real Mem0 extractions, otherwise Llama 3.1 70B.

Taste with headers off and the flat keyword list hold the same text under the same code, so they must rank
alike, and a test holds them to it. They are one result, asked of each reader twice on purpose: what differs between the two readings is
the reader's own noise, reported as the floor under every gap. Any gain they show over no memory belongs
to keyword search, not to version control. Only the header can move Taste's row.

## Data

- **LongMemEval-S**, the cleaned release at revision `98d7416c`, sha256 `d6f21ea9…c3a442`, 500 questions. Dev
  50, held-out 450, grouped so a question stays with its `_abs` twin (seed 20261105).
- **LoCoMo** (`locomo10.json` at commit `3eb6f2c5`, sha256 `79fa87e9…98ff4`), 10 conversations, split whole in
  `data/splits/memory-bench-locomo.json`: 2 dev, 8 held-out. Categories 1-4 are scored; the 446 adversarial
  questions are reported apart.
- **ConPact SWE-Natural** (126 cases): compliance when what a system hands over takes the place of the history, every system
  given the same token cap, read with the benchmark's own probe. Its own compaction is a row beside them; no system's block includes
  a compaction summary. This set has no held-out part: it has no split, its 126 cases are read once, and the cap was set first.
  These logs give no question, so the main table has no query: Mem0 all its memories newest first, MemPalace
  `wake_up`, the flat list and Taste their newest records. A second table, apart, uses the request as the query.
  Each round of a log is one unit. The cap is 1,000 tokens, about what a summary may take: the benchmark's summaries
  run at a median of 694 tokens under the Anthropic prompt and 628 under Pi's, largest 1,253 and 1,129. It was set from
  those sizes before any system's compliance was read, and the same cases at 8,000 tokens are reported beside it. A row with no memory,
  an empty block in place of the history, read by the same stage, was added after the others had been read; it involves no choice.
  These cases were read on earlier code, so the frozen code makes every ranking and block again (`jobs/swe.sbatch`), with the cases exported again and compared byte for byte, and an answer or a verdict stands only if
  its block is the same bytes; those whose block changed are asked again, once (`jobs/swe_read.sbatch`). Of 2,898 answers 58 were: 57 on MemPalace's search, which now asks for 1,000 drawers rather than 200 (the number
  reaches its ranking: the top five change on 11 of 126), and 1 on the keyword list, which now holds the first copy of identical text. Mem0's one ingest of this set stands on the commit it ran on, whose adapter differs only in a docstring, which a test holds.
  Taste's headers for these logs were written first, at an output cap of 1,024 with no fixed temperature; the other two sets' at 800 and temperature 0, from the same prompt. `swe.py report` writes no table unless every row holds every case, each answer read from the block on record and each verdict on it. A cap counts each piece's tokens and not the separators between them, so the words stay within it and a whole block may pass it: by at most 0.5% on most rows and 3.2% on Mem0's many short memories (1,032 at 1,000, 8,198 at 8,000).

## Ingest

The same for every system: oldest first, one fresh store per LongMemEval question and per LoCoMo
conversation, every marker of the answer removed, and neutral ids. Systems that need an environment of their own read one
shared file of what a memory may see, and its hash is recorded. LoCoMo's image captions are in the text for every system,
because the benchmark's reader prompt shows them. MemPalace's own LoCoMo runner leaves them out, so its adapter is checked with
them left out and run with them in.

How a session's date reaches a system differs, and the reader sees every session's date whatever the system. The keyword list,
Taste and the dense baseline index a date line at the head of each unit. MemPalace's benchmark arm ranks what its published runner
ranks, with no date line, so that its lists can be held to the ones it committed, and its product arm gets the time in each line's
own timestamp field. Mem0 gets a system line and the date through `prompt`, since `timestamp` is platform-only.

## Retrieval and the reader

- **Budget**: 8,000 tokens, with a sweep at 2,000, 4,000 and 16,000, filled in each system's own rank order. It counts a memory's
  own words, with one tokenizer (Llama's) for both readers and no heading or role, so that many small pieces pay the same per word as
  a few large ones. An item that does not fit is left out and filling goes on with what still fits, so that one long turn high in a ranking does not end a row; the first
  one of which a part fits is cut between turns, never inside one, and only one item per question is delivered in part. Rows at a reader's window count what the prompt holds, so that
  they fit. Until 2026-11-06 filling stopped at the first item that did not fit. An independent review that day found this cost the rows with large items: on dev at 2,000 tokens the
  keyword list delivered 1,792 on average, and in the held-out part, reading turn sizes alone, 145 of 450 questions hold a round whose first turn is over 2,000 tokens, 15 over 4,000 and 5 over 8,000, where dev holds 6, 0 and 0.
  Each row reports how many questions fell under 90% of its budget.
- **Readers**: Llama 3.1 70B Instruct (revision `1605565b`) and gpt-oss-120b, served by vLLM 0.32.0, both on every row at 8,000
  tokens; the sweep goes to gpt-oss-120b. The two rows with the same prompts, the keyword list and Taste with headers off, are read
  separately on purpose: what differs between them is the reader's own noise, and it is reported as the floor under every gap. Output caps: gpt-oss-120b 4,096
  with reasoning effort low, since its reasoning counts toward the cap, and 12,288 on full context; Llama 800 on LongMemEval, the authors' own for this reading method, and 50 on LoCoMo,
  LoCoMo's own per-answer budget. A reply with no text is asked again, up to eight tries in all, for every row alike, with the tries on record: on full context gpt-oss-120b's reasoning used up its whole cap on 2 of 50 dev questions, and asked three more times each, those two came back with an answer on 4 of 6, so it is chance on questions it is prone to. An answer still empty after eight stays empty, counts as wrong and is counted. Each row reports its empty, cut-off and asked-again answers, and more than 2% empty, or cut off on a reader that reasons first, stops the row.
- **Prompts**: LongMemEval's own reader prompt with its reading method, byte for byte; LoCoMo's own question
  prompts. Never a system's own prompt. What came from one session sits under one heading, in that session's order where a system can name its rounds, and in its rank order where it cannot, as with MemPalace's drawers.

## Scoring

- **Retrieval**: session `recall_any` and `recall_all` at 1, 3, 5, 10, 30 and 50 and at the budget, session NDCG@10, and round
  recall at the budget only, since five whole sessions hold far more rounds than five rounds do. A system that cannot say which round it
  drew on has no round score; one that can, of which nothing whole arrived, reads zero; and for a memory that names its round the score says something drawn from it arrived, not that the fact
  survived. Whether the evidence turn's own words reached the reader is asked only of systems that hand over words as they were said.
  So on LongMemEval every row at 8,000 tokens is also asked whether what it delivered holds enough to answer, by the judge Mem0's makers
  publish for their own retrieval mode (`mem0ai/memory-benchmarks` at `4b61c5d3`, read from that file) on the free served model, and each
  row reports retrieval against the answer as a two-by-two table. Abstention and adversarial questions have no retrieval score.
- **LongMemEval answers**: the official per-type judge prompts, byte for byte. The official judge model
  (`gpt-4o-2024-08-06`) is not available to this study, so gpt-oss-120b, served here at the official temperature of 0, judges every row, and GPT-5.4 on Amazon Bedrock, which takes no
  temperature, reads the answers of the main table's rows, both readers, leaving out the budget sweep and no memory, as a second judge, with agreement and kappa reported, a strict parse, and the official "yes anywhere" rule kept beside it (`paid_judge.py`, AWS credit, on the user's yes of Nov 6: about 9,900 verdicts, about $13 by the 50-answer sample, stopping by itself at $20, in ordinary calls, since Bedrock's batch table lists no GPT-5.4). Every judge reads the question, its date
  and its answer from the same cleaned file the reader's prompt came from.
- **LoCoMo answers**: the benchmark's own token F1, which needs no judge, leads, with the stemmer of its
  evaluation script (NLTK 3.10.4, default Porter stemmer). Adversarial questions are scored by their own rule.
  Every question is scored, then again without the answers a public audit of the gold flags (99 of 1,540,
  13 of them ambiguous; `errors.json` of dial481/locomo-audit at `9493fb4b`, sha256 `f298ef46…`, leaving in its
  57 wrong citations, which fault the evidence and not the answer); the gold is never edited.
- Paired tests per question on LongMemEval, by a sign test with a bootstrap over questions; a bootstrap over conversations on
  LoCoMo. All against Taste with headers off. Gaps under about 4.5 points on LoCoMo are not claimed.

## Differences between systems, all stated

Only MemPalace's hybrid v4 uses the question's date in ranking, as its published runner does. Asking for 1,000 results widens Mem0's and MemPalace's product search beyond their defaults, so that
no budget runs out first. MemPalace's benchmark arms rank what its runner ranks, with no date line. The dense and keyword lists hold one copy of identical text; MemPalace and Mem0 hold what they hold.

## Before any rival's number counts

- MemPalace raw on LongMemEval-S: `recall_any@5` 0.966, `@10` 0.982, NDCG@10 0.889, as published.
- MemPalace on LoCoMo, top 10: session 0.603, dialog 0.480. Its own runner gave 0.60287 and 0.47882 here. The adapter is held to the lists it committed on all ten conversations with captions
  left out, as its runner leaves them: 1,986 of 1,986 top tens alike, raw and hybrid at session level. It committed no list at dialog level, so that arm stands on its runner's number. With captions in, as the study runs it, top tens differ, which is why captions are left out for this check.
- A gap over 0.5 points blocks that system's numbers until it is explained.
- The MemPalace adapter is checked against the per-question ranked lists MemPalace committed at its release.
  This ranks all 500 LongMemEval questions, held-out included, for MemPalace alone and with its published
  settings. No setting is chosen from it, no answer is read, and the held-out run is a separate run.
- The LongMemEval oracle with Llama 3.1 70B: 0.848 published with the official judge, 0.814 here with
  GPT-5.4; the gap is stated, not removed.

## Tuning, freezing and held-out

Each system gets its defaults and one setting its makers recommend, tuned on dev only and with the same
effort. MemPalace's hybrid v4 is also scored, in answers and in retrieval, without the three questions its own notes say it was tuned on (`d6233ab6`, `4dfccbf8`,
`ceb54acb`). Every setting is committed before the held-out part is opened, once, and the stages are the committed job scripts in
`studies/memory_bench/jobs/`. The adapters and a note to each system's maintainers are prepared before the held-out run. Whether
maintainers replied before it ran is recorded.

What needs no gold may be written earlier, since ingest takes a long time on shared GPUs: the shared ingest files, Mem0's ingest, the
headers and each system's ranked lists, which take the question's words alone. Nothing reads those lists and no score is computed until
the part opens on 2026-11-08 at 8 am Eastern; the first stage that reads the gold starts then. If a maintainer's reply changes a system's
settings, that system is run again. The opening was first set for 2026-11-10 at noon, only to leave the maintainers time to answer the note;
the note was never sent, and on 2026-11-08, before any held-out gold was read, the opening was moved to that morning.

Every stage records its commit and a hash of the code on disk, job scripts and split files included. That hash is worked out from git where the repository is and every job holds its copy of the
code to it before it starts, and the stage that first reads the gold refuses to run before the part's date. The long ingest may stand on an earlier commit than the stages after it, as long as what
it depends on has not changed: the adapters, the header writer and their job scripts, by a hash of their own, and the shared ingest files, made again and compared. `studies/memory_bench/analyze.py`
refuses to write any table unless every row named here is there and no other, from one commit and one copy of the code, on the part, budget, system, tokenizer and ranked lists it names, with every
question answered and judged, each answer to the prompt on record, each verdict to the answer on record, and Mem0 and the headers on the settings named here. A row that fails stops the job, so
nothing reads on from it. The dev part rehearses every stage on the commit about to be frozen.

## Dates written into the raw history, a follow-up test (Nov 9)

The held-out comparison found Mem0's lead over plain git concentrated in the date questions. At 8,000 tokens it is all there: 101
against 86 of the 120 date questions, and 259 against 260 on the other 330, with gpt-oss-120b reading. Mem0, as run, is given each
session's date when it stores a memory, and its model writes calendar dates into the facts it keeps. Plain git keeps "yesterday" as
it was said, with the session's date in a header line, so the reader must work the date out. This test asks how much of the lead
comes from that alone.

A pass with no model (`studies/memory_bench/dates.py`) follows each relative date in what people said ("yesterday", "two weeks
ago", "last Friday") with the date it names, worked out from the session's date, and leaves ambiguous ones ("on Friday", "a few
days ago") alone. The system `taste_git_dated` is plain Taste over that history. It was built and frozen on the practice part only,
at commit `bc3fd2d`. The assistant's replies are left as they were, a choice made on 60 sampled practice dates, where people's dates
were right and the assistant's were mostly a reply's "today". On the practice part the dated history delivers the same evidence as
plain git at every budget, with the same tokens, so the test isolates the dates.

Plain git as it was, plain git with dates and Mem0 (its memories from the held-out run) are answered and judged again, together.
Each budget's three rows are read and judged on one server session of gpt-oss-120b, at the held-out run's reading and judging
settings, so that no new row is compared with an old one. Byte-identical prompts change 17 to 35 of 450 verdicts between sessions.
The practice part runs first. The held-out part runs 8,000 tokens first, then 2,000, 4,000 and 16,000, a whole budget at a time,
in the gaps between the long-run sessions on the study's GPU jobs. A budget starts only when its reading and judging fit before the
hold's hard stop. A budget cut by a stop all the same has its partial answers set aside, unread, and is read and judged again whole in
the next hold. The prediction is written after the practice answers and before any held-out answer of this test exists, and
committed before the held-out answers.

**Practice part (50 questions, 13 of them date questions), read Nov 9 by 7:09 pm Eastern:**
- Every row was answered and judged, with none failed.
- Plain git with dates against plain git got 36 against 35, 35 against 36, 37 against 38, and 36 against 38 at 2,000, 4,000, 8,000 and 16,000 tokens.
- On the date questions alone it got 8 against 7, 8 against 10, 9 against 10, and 9 against 9.
- Mem0 got 38, 41, 39 and 42.
- There is no sign either way, on too few questions to say more.

**Prediction for the held-out part, written Nov 9 by 7:11 pm Eastern, before any held-out answer of this test exists (gpt-oss-120b, the main judge):**
1. Plain git with dates against plain git: no difference at any budget after Holm's correction over the four. The point estimate is within 4
   questions of 450 at each budget.
2. On the 120 date questions, dates close at most a quarter of Mem0's lead. The point estimate is 2 questions more than plain git at each budget,
   between 3 fewer and 7 more.
3. On the 330 other questions, no change, within 4 at each budget.
4. Mem0's lead over plain git in this rerun stays as in the held-out run, within the noise between sessions: 14 to 26 on the date questions at
   every budget, and on the others only at 2,000 and 4,000.

Why: the pass dates only explicit relative expressions. Many date questions turn on events told with none ("I just started reading it"),
which Mem0's model dates from the session, and which a raw history leaves to the session's header. If points 1 and 2 hold, Mem0's date
lead needs a model's reading of the event, not date arithmetic; if dates close half its date lead or more, this prediction was wrong.

**Correction, written Nov 9 by 7:55 pm Eastern, after the held-out answers at 2,000 and 8,000 tokens were read.**
- The dated history delivers the same gold evidence as plain git, with recall at the budget within half a point at both budgets. It does not
  deliver the same other pieces, because the written dates change the words and lengths the search ranks by. The pieces delivered match for 296,
  240, 143 and 59 of the 450 questions at 2,000, 4,000, 8,000 and 16,000 tokens.
- The test therefore measures writing dates into the history as a whole, through what the search finds and what the reader sees. It does not
  isolate the dates, as the paragraph above says. The questions whose pieces match exactly, with only the dates added, are reported apart, as
  exploratory: 95 at 2,000 tokens and 94 at 8,000.
- The score `evidence_survived` matches the evidence word for word, which the written dates break, so it is not compared across the two git rows.
- Questions whose two prompts are byte-identical, because no relative date reached the reader, measure the reader's own noise within one
  server session. Their verdicts changed for 11 of 201 at 2,000 tokens and 5 of 52 at 8,000.

## Running on a full cluster

The stages may run as tasks on GPU jobs the study already holds, for when the queue would keep them from their day: the same job scripts and rows, in the same order, kept by marks on the shared disk, so that nothing rests on a login node. A held job ends itself when its GPUs have rested with no work for an hour, or for 8 hours whatever says it is running. Llama reads about 3,000 prompt tokens a second on two GPUs, about 21 hours for this part on one server by the practice run, so a second server takes LoCoMo if one can be had, and on each the rows the comparison turns on go first: Taste with and without headers, the keyword list, Mem0 and MemPalace's product path. On one server LongMemEval goes first, since the second judge needs only those answers. The first reader's server there also carries the tool-call flags the long run needs, which a request with no tools does not use, and one wide window for every stage; the rehearsal runs on the same. On a shared server a reader's cost comes from its answers, and GPU-hours from the record of the job that keeps the GPUs.

## Stopping and record

A run stops at the first surprise in any batch. A model whose tool calls or JSON replies parse less than 95%
of the time stops the arm that depends on it until fixed. Every model call and token, stage by stage and so ingest and reading
apart, goes on one ledger with GPU-hours, by the server's own counters and the record each job keeps of its GPUs. Raw per-question records are kept: ranked and delivered ids, the
prompt's hash, the raw answer and every raw verdict. Counts are checked before any aggregate.
