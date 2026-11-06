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
  for Llama, whose window here is 32,768; 126,000 for gpt-oss, whose window is 131,072 and whose output cap is 4,096): when it
  cuts, the newest sessions are kept, and how many were cut is reported per question; **the newest text**, the same list cut
  to the budget, as a context window would; **LongMemEval oracle** (the evidence sessions only).
- **Flat keyword**: the same units as Taste's store in a plain list, ranked by the same code
  (`taste/memstore/keyword.py`: letters and digits, Porter's stemmer, BM25 with k1 1.5 and b 0.75). Git stores identical
  bytes once, so the list searches the first copy of identical text only.
- **Flat dense**: the same units ranked by cosine similarity under one shared local embedder (chosen on dev).
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
  conversations the date took F1 from .453 to .492 and temporal questions from .280 to .518, with nothing patched, so it is this
  system's setting. `search` with `top_k` 1,000 and `threshold` 0, and each memory goes to the reader as written, in Mem0's rank
  order until the budget is full. At first this asked for 200, which on LoCoMo's short memories ran out at about 6,000 tokens, so dev
  was run again. LoCoMo's `speaker_a` is the user and `speaker_b` the assistant, as in Mem0's harness. gpt-oss-120b reasons inside its
  output cap, so the cap is 4,096 tokens with reasoning effort low: at Mem0's default of 2,000 the reasoning used it up on 84 of 776
  extractions, and at 4,096 all 13,709 practice replies parsed. Mem0's makers recommend an embedder of at least Qwen 600M for hybrid
  search; which embedder its held-out row uses is settled on dev and frozen below, and the choice is stated.

One auxiliary model serves every system that needs one (Mem0's extraction, Taste's headers): gpt-oss-120b if
its JSON replies parse at least 95% of the time on 50 real Mem0 extractions, otherwise Llama 3.1 70B.

Taste with headers off and the flat keyword list hold the same text under the same code, so they must rank
alike. A test holds them to it, and they are reported as one row: any gain they show over no memory belongs
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
  those sizes before any system's compliance was read, and the same cases at 8,000 tokens are reported beside it.

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
  a few large ones. An item that does not fit is cut between turns, never inside one, and filling stops there. Rows at a reader's
  window count what the prompt holds, so that they fit.
- **Readers**: Llama 3.1 70B Instruct (revision `1605565b`) and gpt-oss-120b, served by vLLM 0.32.0, both on every row at 8,000
  tokens; the sweep goes to gpt-oss-120b. The two rows with the same prompts, the keyword list and Taste with headers off, are read
  separately on purpose: what differs between them is the reader's own noise, and it is reported as the floor under every gap.
- **Prompts**: LongMemEval's own reader prompt with its reading method, byte for byte; LoCoMo's own question
  prompts. Never a system's own prompt. What came from one session sits under one heading, in that session's order.

## Scoring

- **Retrieval**: session `recall_any` and `recall_all` at 1, 3, 5, 10, 30 and 50 and at the budget, session NDCG@10, and round
  recall at the budget only, since five whole sessions hold far more rounds than five rounds do. A system that cannot say which round it
  drew on has no round score, and for a memory that names its round the score says something drawn from it arrived, not that the fact
  survived. Whether the evidence turn's own words reached the reader is asked only of systems that hand over words as they were said.
  So on LongMemEval every row at 8,000 tokens is also asked whether what it delivered holds enough to answer, by the judge Mem0's makers
  publish for their own retrieval mode (`mem0ai/memory-benchmarks` at `4b61c5d3`, read from that file) on the free served model, and each
  row reports retrieval against the answer as a two-by-two table. Abstention and adversarial questions have no retrieval score.
- **LongMemEval answers**: the official per-type judge prompts, byte for byte. The official judge model
  (`gpt-4o-2024-08-06`) is not available to this study, so GPT-5.4 and gpt-oss-120b on Amazon Bedrock judge,
  with agreement and kappa reported, a strict parse, and the official "yes anywhere" rule kept beside it.
- **LoCoMo answers**: the benchmark's own token F1, which needs no judge, leads, with the stemmer of its
  evaluation script (NLTK 3.10.4, default Porter stemmer). Adversarial questions are scored by their own rule.
  Every question is scored, then again without the answers a public audit of the gold flags (99 of 1,540,
  13 of them ambiguous; `errors.json` of dial481/locomo-audit at `9493fb4b`, sha256 `f298ef46…`, leaving in its
  57 wrong citations, which fault the evidence and not the answer); the gold is never edited.
- Paired tests per question on LongMemEval, by a sign test with a bootstrap over questions; a bootstrap over conversations on
  LoCoMo. All against Taste with headers off. Gaps under about 4.5 points on LoCoMo are not claimed.

## Before any rival's number counts

- MemPalace raw on LongMemEval-S: `recall_any@5` 0.966, `@10` 0.982, NDCG@10 0.889, as published.
- MemPalace on LoCoMo, top 10: session 0.603, dialog 0.480.
- A gap over 0.5 points blocks that system's numbers until it is explained.
- The MemPalace adapter is checked against the per-question ranked lists MemPalace committed at its release.
  This ranks all 500 LongMemEval questions, held-out included, for MemPalace alone and with its published
  settings. No setting is chosen from it, no answer is read, and the held-out run is a separate run.
- The LongMemEval oracle with Llama 3.1 70B: 0.848 published with the official judge, 0.814 here with
  GPT-5.4; the gap is stated, not removed.

## Tuning, freezing and held-out

Each system gets its defaults and one setting its makers recommend, tuned on dev only and with the same
effort. MemPalace's hybrid v4 is also scored without the three questions its own notes say it was tuned on (`d6233ab6`, `4dfccbf8`,
`ceb54acb`). Every setting is committed before the held-out part is opened, once, and the stages are the committed job scripts in
`studies/memory_bench/jobs/`. The adapters and a note to each system's maintainers are prepared before the held-out run. Whether
maintainers replied before it ran is recorded.

What needs no gold may be written earlier, since ingest takes a long time on shared GPUs: the shared ingest files, Mem0's ingest, the
headers and each system's ranked lists, which take the question's words alone. Nothing reads those lists and no score is computed until
the part opens on 2026-11-10 at noon Eastern; the first stage that reads the gold starts then. If a maintainer's reply changes a system's
settings, that system is run again.

Every stage records its commit and a hash of the code on disk. `studies/memory_bench/analyze.py` refuses to write any table unless every
row came from one commit and one copy of the code, on the part and budget it names, with every question answered and judged, each
answer to the prompt on record, and Mem0 and the headers on the settings named here.

## Stopping and record

A run stops at the first surprise in any batch. A model whose tool calls or JSON replies parse less than 95%
of the time stops the arm that depends on it until fixed. Every model call and token, ingest and reading
apart, goes on one ledger with GPU-hours. Raw per-question records are kept: ranked and delivered ids, the
prompt's hash, the raw answer and every raw verdict. Counts are checked before any aggregate.
