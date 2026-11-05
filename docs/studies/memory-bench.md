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

- **No memory**; **full context** (when the budget cuts, the newest sessions are kept, and how many were cut is
  reported per question; every prompt is checked to fit the reader's window); **LongMemEval oracle** (the
  evidence sessions only).
- **Flat keyword**: the same units as Taste's store in a plain list, ranked by the same code
  (`taste/memstore/keyword.py`: letters and digits, Porter's stemmer, BM25 with k1 1.5 and b 0.75).
- **Flat dense**: the same units ranked by cosine similarity under one shared local embedder (chosen on dev).
- **Taste**: `taste.memstore`, a fresh store per question, one file per round, one commit per session, searched
  over contents with `Store.search_contents`; once with headers off, once with headers on. A header holds a
  one-sentence "what" and keywords written by the one auxiliary model from answer-free text, plus fields the
  commit knows itself.
- **MemPalace 3.11.0** (release commit `602d796c`): its own benchmark runner in raw mode, and its product path
  (mining and search), reported apart.
- **Mem0 open source 2.2.1**: a locally served model, a local embedder and a local vector store, with no
  route to any paid service.

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
- **ConPact SWE-Natural** (126 cases): compliance after one compaction, every system given the same token cap
  and no query.

## Ingest

The same for every system: oldest first, one fresh store per LongMemEval question and per LoCoMo
conversation, every marker of the answer removed, neutral ids, and the same date line at the head of each
unit. Systems that need an environment of their own read one shared file of what a memory may see, and its
hash is recorded.

## Retrieval and the reader

- **Budget**: 8,000 tokens, with a sweep at 2,000, 4,000 and 16,000, counted with the reader's tokenizer and
  filled in each system's own rank order. An item that does not fit is cut between turns, never inside one,
  and filling stops there.
- **Readers**: Llama 3.1 70B Instruct (revision `1605565b`) and gpt-oss-120b, served by vLLM 0.32.0.
- **Prompts**: LongMemEval's own reader prompt with its reading method, byte for byte; LoCoMo's own question
  prompts. Never a system's own prompt.

## Scoring

- **Retrieval**: session and round `recall_any` and `recall_all` at 1, 3, 5, 10, 30 and 50 and at the budget,
  session NDCG@10, and whether the evidence turn's own words reached the reader. Abstention questions have
  no retrieval score.
- **LongMemEval answers**: the official per-type judge prompts, byte for byte. The official judge model
  (`gpt-4o-2024-08-06`) is not available to this study, so GPT-5.4 and gpt-oss-120b on Amazon Bedrock judge,
  with agreement and kappa reported, a strict parse, and the official "yes anywhere" rule kept beside it.
- **LoCoMo answers**: the benchmark's own token F1, which needs no judge, leads, with the stemmer of its
  evaluation script (NLTK 3.10.4, default Porter stemmer). Adversarial questions are scored by their own rule.
  Every question is scored, then again without the answers a public audit of the gold flags (99 of 1,540,
  13 of them ambiguous); the gold is never edited.
- Paired tests per question on LongMemEval; a bootstrap over conversations on LoCoMo. Gaps under about 4.5
  points on LoCoMo are not claimed.

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
effort. Every setting is committed before the held-out part is opened, once; the run refuses otherwise. The
adapters and a note to each system's maintainers are prepared before the held-out run. Whether maintainers
replied before it ran is recorded.

## Stopping and record

A run stops at the first surprise in any batch. A model whose tool calls or JSON replies parse less than 95%
of the time stops the arm that depends on it until fixed. Every model call and token, ingest and reading
apart, goes on one ledger with GPU-hours. Raw per-question records are kept: ranked and delivered ids, the
prompt's hash, the raw answer and every raw verdict. Counts are checked before any aggregate.
