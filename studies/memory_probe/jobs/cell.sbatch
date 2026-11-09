#!/bin/bash
#SBATCH --job-name=aos-probe-cell
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# One cell of the rule-loss matrix: a case set (IN), a summary prompt (PR) and a depth (K), every arm, on the settings frozen in the protocol. Notes and headers are written
# once per case set, in OUT, and any job can be run again and carries on, so long as what its rows read has not changed: they refuse otherwise. ROWS names the rows to run, all of them
# when empty. A row that fails, or a judge, fails the job. Free. Needs STUDY_ROOT and SHA.
set -uo pipefail
unset ANTHROPIC_API_KEY OPENAI_API_KEY
D=${STUDY_ROOT:?}; S=${SHA:?}
IMG=$D/images/vllm-v0.32.0.sif
CODE=$D/code-${S:0:12}
PR=${PR:?}; K=${K:?}; IN=${IN:?}; OUT=${OUT:?}; NOTES_FROM=${NOTES_FROM:-}; HEADERS_FROM=${HEADERS_FROM:-}; BUDGET=${BUDGET:-4096}; LABEL=${LABEL:-}; PROCS=${PROCS:-28}; FAILED=0
JOB=$OUT/job-$SLURM_JOB_ID; mkdir -p $JOB
export APPTAINER_CACHEDIR=$D/cache/apptainer APPTAINER_TMPDIR=$D/tmp HF_HOME=$D/cache/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR=$D/cache/tiktoken VLLM_CACHE_ROOT=$D/cache/vllm XDG_CACHE_HOME=$D/cache/xdg PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TASTE_COMMIT=$S
export GIT_AUTHOR_NAME=memory-probe GIT_AUTHOR_EMAIL=memory-probe@localhost GIT_COMMITTER_NAME=memory-probe GIT_COMMITTER_EMAIL=memory-probe@localhost
PY=$D/venv-probe/bin/python
RUN="apptainer exec --home $D/cache/home --bind $D $IMG"
MODEL=openai/gpt-oss-120b
fail() { echo "FAILED: $*"; kill ${PID:-} ${SMI:-} 2>/dev/null; exit 1; }
# The agent: the model whose rows these are, which writes the notes and the told summary and answers every probe. A model other than the study's
# gpt-oss-120b answers the same cases with the headers the memory's own writer made (HEADERS_FROM), on a server already up (SERVER_URL), and its rows are
# judged apart on gpt-oss-120b (JUDGE=later), so that only the agent changes. AGENT_EFFORT "none" sends no reasoning effort; AGENT_CAP sets the answer,
# notes and summary caps, for a model that thinks at length.
# Another agent's writers are kept to writing (KEEP, READS): Laguna S 2.1 carried on the work in 105 of its 360 notes versions on the matrix's small test (Nov 9),
# so a notes reply that works is asked again with the long run's sentence last, and a told summary is read by the long run's rule. The study model's run as they did.
AGENT_MODEL=${AGENT_MODEL:-$MODEL}; AGENT_EFFORT=${AGENT_EFFORT:-low}; AGENT_CAP=${AGENT_CAP:-}; KEEP=(); READS=()
if [ "$AGENT_MODEL" != "$MODEL" ]; then
  [ -n "${SERVER_URL:-}" ] || fail "another agent needs SERVER_URL"
  { [ -n "$HEADERS_FROM" ] && [ -e "$HEADERS_FROM/turn-headers.jsonl" ]; } || fail "another agent keeps the memory's own headers: HEADERS_FROM"
  [ "${JUDGE:-}" = later ] || fail "another agent's rows are judged apart, on gpt-oss-120b: JUDGE=later"
  KEEP=(--keep-to-notes); READS=(--read-summaries)
fi
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT cell $PR k$K budget $BUDGET in $IN out $OUT"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 > $JOB/gpu.csv &
SMI=$!
# SERVER_URL: use a server another of this study's jobs is already running with the same model and settings, on rows that offer no tools.
if [ -n "${SERVER_URL:-}" ]; then
  URL=$SERVER_URL; PID=
  curl -sf ${URL%/v1}/health >/dev/null || fail "no server at $URL"
  echo "using the running server at $URL"
else
  PORT=$((20000 + RANDOM % 20000)); URL=http://127.0.0.1:$PORT/v1; T0=$(date +%s)
  apptainer exec --nv --home $D/cache/home --bind $D $IMG vllm serve $D/models/gpt-oss-120b --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
    --max-model-len 131072 --gpu-memory-utilization 0.92 --enable-auto-tool-choice --tool-call-parser openai > $JOB/vllm.log 2>&1 &
  PID=$!
  until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
    sleep 5; kill -0 $PID 2>/dev/null || { tail -40 $JOB/vllm.log; fail "server died"; }
    [ $(( $(date +%s) - T0 )) -gt 1800 ] && fail "load timeout"
  done
  echo "load_seconds $(( $(date +%s) - T0 ))"
fi
cd $CODE
TOLD=$OUT/notes-told.jsonl; NEUTRAL=$OUT/notes-neutral.jsonl; HEADERS=$OUT/turn-headers.jsonl
# Notes written earlier carry over unless a version came back empty: those cases are written again.
if [ -n "$NOTES_FROM" ] && [ ! -e $TOLD ]; then
  $PY - $NOTES_FROM $OUT <<'PY'
import json, sys
from pathlib import Path
source, out = Path(sys.argv[1]), Path(sys.argv[2])
for name in ("notes-told.jsonl", "notes-neutral.jsonl"):
    rows = [json.loads(line) for line in (source / name).read_text().splitlines()]
    kept = [row for row in rows if all(version.strip() for version in row["versions"])]
    (out / name).write_text("".join(json.dumps(row) + "\n" for row in kept))
    print(name, "kept", len(kept), "of", len(rows))
PY
fi
[ -n "$HEADERS_FROM" ] && [ ! -e $HEADERS ] && cp $HEADERS_FROM/turn-headers.jsonl $HEADERS
# WRITERS=0: a cued case has the published case's history word for word, so it keeps that case's notes and headers, read by the history they came from.
wanted() { [ -z "${ROWS:-}" ] || [[ " $ROWS " == *" $1 "* ]]; }
needs() { [ -z "${ROWS:-}" ] || [[ "$ROWS" =~ $1 ]]; }  # a pattern over the rows asked for
if [ "${WRITERS:-1}" = 1 ]; then
  # A set's notes and headers are written once, by whichever of its cells gets here first: another waits here, and finds them written. Each writer asks only what it lacks, so
  # it is run again here, before any row has read its file: a row started on a file that later changes refuses to carry on.
  exec 9> $OUT/.writers.lock; flock 9
  if needs notes-; then
    for style in told neutral; do
      T1=$(date +%s)
      for try in 1 2 3; do
        $PY -m studies.memory_probe.notes --inputs $IN --url $URL --model $AGENT_MODEL --effort $AGENT_EFFORT ${AGENT_CAP:+--max-tokens $AGENT_CAP} ${KEEP[@]+"${KEEP[@]}"} --style $style --workers 32 --out $OUT/notes-$style.jsonl > $JOB/notes-$style.log 2>&1
        code=$?; [ $code = 0 ] && break
      done
      [ $code = 0 ] || FAILED=$((FAILED + 1))
      echo "notes $style exit $code tries $try seconds $(( $(date +%s) - T1 )) $(tail -1 $JOB/notes-$style.log | cut -c1-200)"
    done
  fi
  if needs "headers|rules-and-search|header-list" && [ "$AGENT_MODEL" = "$MODEL" ]; then
    T1=$(date +%s)
    for try in 1 2 3; do
      $PY -m studies.memory_probe.headers --inputs $IN --url $URL --model $MODEL --out $HEADERS --workers 32 > $JOB/headers.log 2>&1
      code=$?; [ $code = 0 ] && break
    done
    [ $code = 0 ] || FAILED=$((FAILED + 1))
    echo "headers exit $code tries $try seconds $(( $(date +%s) - T1 )) $(tail -1 $JOB/headers.log | cut -c1-200)"
  fi
  flock -u 9
fi
one() {  # label arm condition extra...
  local label=$1 arm=$2 condition=$3; shift 3
  wanted $label || return 0
  local name=$arm-$condition-$PR-k$K-b$BUDGET T1=$(date +%s) row=$OUT/$label/$arm-$condition-$PR-k$K-b$BUDGET PIDS=() I bad=0 code
  mkdir -p $row
  for I in $(seq 0 $((PROCS-1))); do
    $PY -m studies.memory_probe.run --inputs $IN --prompt $PR --k $K --arm $arm --condition $condition --budget $BUDGET --shard $I/$PROCS --workers 1 \
        --url $URL --model $AGENT_MODEL --effort $AGENT_EFFORT ${AGENT_CAP:+--max-tokens $AGENT_CAP} --tokenizer $D/models/gpt-oss-120b/tokenizer.json \
        --workdir /tmp/aos-probe-$SLURM_JOB_ID-$$-$label-$name --out $row/shard$I.jsonl "$@" > $row/shard$I.log 2>&1 &
    PIDS+=($!)
  done
  : > $row/run.log
  for I in $(seq 0 $((PROCS-1))); do
    wait ${PIDS[$I]} || { bad=$((bad + 1)); tail -2 $row/shard$I.log >> $row/run.log; }
  done
  if [ $bad = 0 ]; then
    $PY -m studies.memory_probe.merge --inputs $IN --out $row/results.jsonl $row/shard*[0-9].jsonl >> $row/run.log 2>&1
    code=$?
  else
    code="$bad-shards-failed"
  fi
  [ "$code" = 0 ] || FAILED=$((FAILED + 1))
  echo "$label $name exit $code seconds $(( $(date +%s) - T1 )) $(tail -1 $row/run.log | cut -c1-300)"
  rm -rf /tmp/aos-probe-$SLURM_JOB_ID-$$-$label-$name
}
judge() {  # label arm
  wanted $1 || return 0
  [ "${JUDGE:-}" = later ] && return 0
  local results=$OUT/$1/$2-i-$PR-k$K-b$BUDGET/results.jsonl said code
  said=$($RUN $D/venv/bin/python -m studies.memory_probe.judge --results $results --cases $IN/cases.jsonl --url $URL --model $MODEL 2>&1); code=$?
  [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "judge $1 exit $code $(echo "$said" | tail -1 | cut -c1-200)"
}
judge_summary() {  # label summaries: a summary row's block is its summary, so that is what the judge reads
  wanted $1 || return 0
  [ "${JUDGE:-}" = later ] && return 0
  local results=$OUT/$1/a1-i-$PR-k$K-b$BUDGET/results.jsonl said code
  said=$($RUN $D/venv/bin/python -m studies.memory_probe.judge --results $results --summaries $2 --prompt $PR --k $K --cases $IN/cases.jsonl --url $URL --model $MODEL 2>&1); code=$?
  [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "judge $1 exit $code $(echo "$said" | tail -1 | cut -c1-200)"
}
stored() {  # the wording of the note-taker: what its notes hold at all
  [ "${JUDGE:-}" = later ] && return 0
  local said code
  said=$($RUN $D/venv/bin/python -m studies.memory_probe.judge --notes $OUT/notes-$1.jsonl --out $OUT/stored-$1.served.jsonl --cases $IN/cases.jsonl --url $URL --model $MODEL 2>&1); code=$?
  [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "stored $1 exit $code $(echo "$said" | tail -1 | cut -c1-200)"
}
one summary a1 i
one restated a1 i --upper
one flat-newest a2a i
one rules a6 i
one rules-in-headers a6h i
one turns a5q i
one turns-headers a4q i --headers $HEADERS
one rules-and-search a7q i --headers $HEADERS
for style in told neutral; do
  N=$OUT/notes-$style.jsonl
  one notes-last-$style a2cq i --notes $N
  one notes-every-$style a2dq i --notes $N
  one notes-newest-$style a2cw i --notes $N
  one notes-newest-and-older-$style a2dw i --notes $N
done
if [ "${SMALL:-0}" = 0 ]; then
  one header-list a3 i --headers $HEADERS
  one plain-nothing a5 vi
  one plain-exists a5 vii
  one plain-check a5 viii
fi
judge_summary summary $IN/summaries.jsonl
judge flat-newest a2a; judge turns a5q; judge turns-headers a4q; judge rules a6; judge rules-in-headers a6h; judge rules-and-search a7q
[ "${SMALL:-0}" = 0 ] && judge header-list a3
for style in told neutral; do
  judge notes-last-$style a2cq; judge notes-every-$style a2dq; judge notes-newest-$style a2cw; judge notes-newest-and-older-$style a2dw
  [ "${WRITERS:-1}" = 1 ] && needs notes- && stored $style
done
# The summary told to keep what the user asked it to keep doing, on the same prompt, so that the store and the instruction can be read apart. A cued set has the published
# case's history word for word, so its summaries are the published case's too, and this row belongs to that set.
if [ "$K" = 1 ] && [ "$BUDGET" = 4096 ] && [ "${WRITERS:-1}" = 1 ] && wanted summary-told; then
  T1=$(date +%s)
  $RUN $D/venv/bin/python -m studies.memory_probe.told_summary --inputs $IN --prompt $PR --url $URL --model $AGENT_MODEL --effort $AGENT_EFFORT ${AGENT_CAP:+--output-cap $AGENT_CAP} ${READS[@]+"${READS[@]}"} --out $OUT/told-$PR > $JOB/told-$PR.log 2>&1
  code=$?; [ $code = 0 ] || FAILED=$((FAILED + 1))
  echo "told summaries exit $code seconds $(( $(date +%s) - T1 )) $(tail -1 $JOB/told-$PR.log | cut -c1-200)"
  PUBLISHED=$IN; IN=$OUT/told-$PR
  one summary-told a1 i
  judge_summary summary-told $OUT/told-$PR/summaries.jsonl
  IN=$PUBLISHED
fi
curl -s ${URL%/v1}/metrics | grep -E "^vllm:(prompt_tokens_total|generation_tokens_total|prefix_cache_(hits|queries)_total|request_success_total)" > $JOB/vllm-metrics.txt
kill $PID $SMI 2>/dev/null; [ -n "$PID" ] && wait $PID 2>/dev/null
echo "end $(date -u +%FT%TZ) rows failed $FAILED"
[ $FAILED = 0 ]
