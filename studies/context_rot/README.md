# Context rot: reproduction runs

Before a memory method is tested on two published benchmarks, their own baselines are run
on our setup with their models and scoring, and must land near their published numbers.

| File | What it does |
|---|---|
| `bedrock.py` | One-question Bedrock Converse client for the judges. Token from `AWS_BEARER_TOKEN_BEDROCK`, never printed |
| `lme_judge.py` | LongMemEval's judge prompts, byte for byte, asked of GPT-5.4 or gpt-oss-120b on Bedrock; writes the official `.eval-results-<judge>` file |
| `lme_metrics.py` | LongMemEval's aggregates without its GPT-4o-only assertion, plus unparseable and missing counts, and agreement between two judges |
| `lme_clean.py` | Removes every answer marker from LongMemEval before a memory under test sees it |
| `conpact_bridge.py` | Runs the conpact package as published, with its retention judge asked through Bedrock, and reports what its summary drops |

Outputs go under `studies/context_rot/runs/` and inputs under `studies/context_rot/data/`,
both ignored by git.

## Judges

The papers' judges cannot be used as published: LongMemEval's is `gpt-4o-2024-08-06` through
OpenAI, and conpact's is GPT-5.4 through OpenAI. This study has no OpenAI account.

- Rule-loss benchmark: the same GPT-5.4, through Bedrock (`global.openai.gpt-5.4`), with their
  prompt and their YES/NO rule. Same model, different service.
- LongMemEval: a substitute. GPT-5.4 and gpt-oss-120b (`openai.gpt-oss-120b-1:0`) both judge
  every hypothesis with the official prompts; both are reported with their agreement. The
  paper's lenient substring rule is recorded beside a strict one-word parse. Comparability with
  the published numbers rests on reproducing a published baseline (below), not on the judge
  being the same.
- Neither judge is sent a temperature (the Bedrock reasoning models refuse it beside an
  effort); both run at low reasoning effort with a 2,048-token cap.

Smoke check, Nov 3 2026: eight calls on cases with obvious verdicts, all correct, about $0.004.

## Run 1: rule-loss benchmark, 75 cases

Agent-trace set (Hermes), first 5 histories x 15 rules, gpt-oss-120b as summarizer and as
the model that answers, both Pi and Anthropic summary prompts, GPT-5.4 judge on Bedrock.

1. On the GPU cluster, serve gpt-oss-120b with vLLM on one RTX PRO 6000, as the paper did
   (`vllm serve openai/gpt-oss-120b --max-model-len 131072 --port 8000`). One Hermes history
   is 130,355 tokens; check the first five's `token_length` before running, since a compaction
   prompt over the window fails the whole package run.
2. From the workstation that holds the Bedrock token, through an SSH tunnel, so the token never
   reaches the cluster (the bridge only sends requests; the GPU does the work):

   ```
   ssh -N -L 8000:<gpu-node>:8000 <user>@<cluster-login> &
   set -a; source <(grep '^AWS_BEARER_TOKEN_BEDROCK=' .env); set +a
   # conpact==0.1.0 with openai, datasets, tqdm, in a separate virtual environment
   python -m studies.context_rot.conpact_bridge --source hermes --contexts 5 --prompt pi_mono \
       --probe-url http://localhost:8000/v1 --out studies/context_rot/runs/conpact/hermes-pi
   python -m studies.context_rot.conpact_bridge --source hermes --contexts 5 --prompt anthropic \
       --probe-url http://localhost:8000/v1 --out studies/context_rot/runs/conpact/hermes-anthropic
   ```

Published, Hermes, gpt-oss-120b, N = 750 (their Table 8): kept 36.3% (Pi) and 18.5%
(Anthropic); compacted compliance 60.7 and 58.5; restated ceiling 98.4 and 98.3; effective
retention 27.8 and 23.8; full history 65.1 with the rule and 46.1 without.

At 75 cases from 5 histories the noise is about plus or minus 10 points on retention, so this
run checks that every stage works and lands in the right range; matching within about 3 points
needs all 750. Paid: about 150 judge calls (two prompts x 75), well under $1.

## Run 2: LongMemEval, only the needed sessions (oracle), all 500 questions

Reader Llama 3.1 70B Instruct, the paper's note-taking prompt with JSON history, judged by both
Bedrock judges.

1. On the GPU cluster, with Hugging Face access to the gated Llama 3.1 weights, serve the reader under the
   name LongMemEval's scripts ask for (`run_generation.sh` maps `llama-3.1-70b-instruct` to
   `meta-llama/Meta-Llama-3.1-70B-Instruct`, the repository's former name):

   ```
   vllm serve meta-llama/Llama-3.1-70B-Instruct --served-model-name meta-llama/Meta-Llama-3.1-70B-Instruct \
       --tensor-parallel-size 2 --max-model-len 128000 --port 8001
   ```

   Two RTX PRO 6000 (96 GB each) hold the bf16 weights; the reader script also loads the model's
   tokenizer from Hugging Face, so it needs the same access.
2. Generate with the paper's own script (LongMemEval commit 9e0b455):

   ```
   wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
   cd src/generation && bash run_generation.sh ../../data/longmemeval_oracle.json \
       llama-3.1-70b-instruct full-history-session 1000 json false con
   ```

   Check the hypothesis file has 500 lines: the script skips failed calls without saying so.
3. Judge from the workstation that holds the token:

   ```
   python -m studies.context_rot.lme_judge <hypotheses.jsonl> longmemeval_oracle.json --judge gpt-5.4
   python -m studies.context_rot.lme_judge <hypotheses.jsonl> longmemeval_oracle.json --judge gpt-oss-120b
   python -m studies.context_rot.lme_metrics <hypotheses.jsonl>.eval-results-gpt-5.4 longmemeval_oracle.json \
       --other <hypotheses.jsonl>.eval-results-gpt-oss-120b
   ```

Published, oracle, Llama 3.1 70B: 0.848 with note-taking and JSON history (their Figures 3b and
6 agree), 0.744 answering directly. The oracle file is byte-identical to the paper's. At 500
questions the noise is about plus or minus 3 points. Paid: 1,000 judge calls, about $1.50.

## Before a memory method is run

- The memory gets only what `lme_clean.py` leaves, and a fresh store per question.
- In the rule-loss benchmark the method gets only the history (never the rule fields), returns
  at most the baselines' summary budget (500 tokens), builds its context before the question,
  and returns it as earlier messages.
