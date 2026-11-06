#!/bin/bash
#SBATCH --job-name=aos-part-swe
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
# ConPact's coding set on this commit, with no model called: the cases exported again and held to the first run's, every system's rankings made again from the shared ingest, Taste's
# headers and Mem0's one ingest taken from that run, the blocks at both caps, and the answers and verdicts on record carried over wherever this code makes the same block byte for byte.
# What is left is asked by swe_read.sbatch. FROM is the first run, FROM_ANSWERS and FROM_JUDGED its answers and verdicts. Free.
set -uo pipefail
export PART=swe
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
FROM=${FROM:?the first run}; FROM_ANSWERS=${FROM_ANSWERS:?its answers}; FROM_JUDGED=${FROM_JUDGED:?its verdicts}
case "$FROM_ANSWERS $FROM_JUDGED" in "$FROM"/*" $FROM"/*) ;; *) fail "the answers and verdicts carried over are not from the run named" ;; esac
mkdir -p $R/cases $R/rankings $R/blocks $R/answers $R/judged $R/baselines $R/headers
export MEMPALACE_CONFIG_DIR=$R/mempalace-config MEMPALACE_EMBEDDING_MODEL=minilm MEMPALACE_EMBEDDING_DEVICE=cpu MEMPALACE_EMBEDDING_THREADS=2
mkdir -p $MEMPALACE_CONFIG_DIR
WORK=/tmp/aos-swe-$SLURM_JOB_ID
trap 'rm -rf "$WORK"' EXIT
TOK=$OSS/tokenizer.json; C=$R/cases/cases.jsonl; CASES=$R/cases/views.jsonl; FAILED=0
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT"
cd $CODE
$INSIDE $D/venv/bin/python -m studies.memory_bench.swe export --out $R/cases | cut -c1-200
[ "${PIPESTATUS[0]}" = 0 ] || fail "export"
cmp $C $FROM/cases/cases.jsonl && cmp $CASES $FROM/cases/views.jsonl || fail "the cases are not the first run's"
cp $FROM/headers/swe.jsonl $R/headers/swe.jsonl || fail "headers"
cp $FROM/rankings/mem0f_free.jsonl $FROM/rankings/mem0f_search.jsonl $R/rankings/ || fail "Mem0's rankings"
cp ${FROM_BASELINES:-$FROM_ANSWERS/..}/baseline-*.report.json $R/baselines/ || fail "the benchmark's own rows"
finished() {  # label start
  cat $R/rankings/$1.shard*[0-9].jsonl > $R/rankings/$1.jsonl
  echo "$1: $(wc -l < $R/rankings/$1.jsonl) cases, $(( $(date +%s) - $2 )) s"
  local log
  for log in $R/rankings/$1.shard*.log; do grep -q '"ran"' $log || { FAILED=$((FAILED + 1)); echo "  DID NOT FINISH $log"; grep -v pthread $log | tail -3 | cut -c1-240; }; done
}
ours() {  # label options
  local label=$1 t0=$(date +%s) PIDS=() I; shift
  for I in $(seq 0 15); do
    $PY -m studies.memory_bench.swe rank --views $CASES --workdir $WORK --shard $I/16 --out $R/rankings/$label.shard$I.jsonl "$@" > $R/rankings/$label.shard$I.log 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"; finished $label $t0
}
palace() {  # label options
  local label=$1 t0=$(date +%s) PIDS=() I; shift
  for I in $(seq 0 15); do
    $INSIDE $B/venv-mempalace/bin/python -m studies.memory_bench.adapters.mempalace_arms --views $CASES --repo $B/src/mempalace --arm product \
      --workdir $WORK --shard $I/16 --out $R/rankings/$label.shard$I.jsonl "$@" > $R/rankings/$label.shard$I.log 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"; finished $label $t0
}
ours flat_free --system flat --query-free
ours taste_free --system taste --query-free
ours flat_search --system flat
ours taste_search --system taste
ours taste_headers_free --system taste --headers $R/headers/swe.jsonl --query-free
ours taste_headers_search --system taste --headers $R/headers/swe.jsonl
palace mempalace_free --query-free
palace mempalace_search
T0=$(date +%s); PIDS=()
for I in 0 1 2 3; do
  $INSIDE $B/venv-mem0/bin/python -m studies.memory_bench.adapters.flat_dense --views $CASES --embedder sentence-transformers/all-MiniLM-L6-v2 \
    --shard $I/4 --out $R/rankings/dense_search.shard$I.jsonl > $R/rankings/dense_search.shard$I.log 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"; finished dense_search $T0
$PY -m studies.memory_bench.swe none --cases $C --out $R/rankings/none.jsonl || fail "the row with no memory"
row() {  # label cap
  local name=$1-$2
  $PY -m studies.memory_bench.swe blocks --rankings $R/rankings/$1.jsonl --cases $C --tokenizer $TOK --cap $2 --out $R/blocks/$name.jsonl > /dev/null || { FAILED=$((FAILED + 1)); echo "$name: no blocks"; return; }
  local JUDGED=(--judged $FROM_JUDGED/$name --judged-out $R/judged/$name)
  [ "$1" = none ] && JUDGED=()
  LINE=$($PY -m studies.memory_bench.swe carry --blocks $R/blocks/$name.jsonl --answers $FROM_ANSWERS/$name.jsonl --out $R/answers/$name.jsonl ${JUDGED[@]+"${JUDGED[@]}"} 2>&1) || FAILED=$((FAILED + 1))
  echo "$name $LINE" | cut -c1-300
}
row none 1000
for LABEL in flat_free taste_free taste_headers_free mem0f_free mempalace_free flat_search taste_search taste_headers_search dense_search mem0f_search mempalace_search; do
  for CAP in 1000 8000; do row $LABEL $CAP; done
done
echo "end $(date -u +%FT%TZ) failed $FAILED"
[ $FAILED = 0 ]
