#!/bin/bash
#SBATCH --job-name=aos-part-rank
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=10:00:00
# The shared ingest files, MemPalace in all its arms and the dense baseline: ranked lists from the question's words, with no model and no gold.
# On the held-out part nothing reads these until the part opens. Free.
set -uo pipefail
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
mkdir -p $R $V
export MEMPALACE_CONFIG_DIR=$R/mempalace-config MEMPALACE_EMBEDDING_MODEL=minilm MEMPALACE_EMBEDDING_DEVICE=cpu MEMPALACE_EMBEDDING_THREADS=2
mkdir -p $MEMPALACE_CONFIG_DIR
step() { echo; echo "=== $* ($(date -u +%T))"; }
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT part $PART"
cd $CODE
step "the shared ingest files"
$PY -m studies.memory_bench.run views --benchmark lme --data $LME_DATA --split $LME_SPLIT --part $PART --out $V/lme.jsonl || fail "views lme"
$PY -m studies.memory_bench.run views --benchmark locomo --data $LOC_DATA --split $LOC_SPLIT --part $PART --out $V/locomo.jsonl || fail "views locomo"
arm_mempalace() {  # view arm processes
  local view=$1 arm=$2 n=$3 out=$R/rankings/$1/mempalace_$2 t0=$(date +%s) PIDS=()
  mkdir -p $out
  for I in $(seq 0 $((n-1))); do
    $RUN $B/venv-mempalace/bin/python -m studies.memory_bench.adapters.mempalace_arms --views $V/$view.jsonl --repo $B/src/mempalace \
      --arm $arm --workdir /tmp/aos-mp-$SLURM_JOB_ID --shard $I/$n --out $out/shard$I.jsonl > $out/shard$I.log 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  cat $out/shard*[0-9].jsonl > $out/rankings.jsonl
  echo "$view mempalace_$arm: $(wc -l < $out/rankings.jsonl) questions, $(( $(date +%s) - t0 )) s"
  for I in $(seq 0 $((n-1))); do grep -q '"ran"' $out/shard$I.log || { echo "  SHARD $I DID NOT FINISH:"; grep -v pthread $out/shard$I.log | tail -3 | cut -c1-240; }; done
}
step "MemPalace on LongMemEval-S"
for ARM in bench_raw bench_v4 product; do arm_mempalace lme $ARM 16; done
step "MemPalace on LoCoMo"
for ARM in locomo_raw_session locomo_raw_dialog locomo_hybrid_session product; do arm_mempalace locomo $ARM 8; done
step "the dense baseline"
for VIEW in lme locomo; do
  out=$R/rankings/$VIEW/flat_dense; mkdir -p $out; t0=$(date +%s); PIDS=()
  for I in $(seq 0 7); do
    $RUN $B/venv-mem0/bin/python -m studies.memory_bench.adapters.flat_dense --views $V/$VIEW.jsonl \
      --embedder sentence-transformers/all-MiniLM-L6-v2 --shard $I/8 --out $out/shard$I.jsonl > $out/shard$I.log 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  cat $out/shard*[0-9].jsonl > $out/rankings.jsonl
  echo "$VIEW flat_dense: $(wc -l < $out/rankings.jsonl) questions, $(( $(date +%s) - t0 )) s"
  for f in $out/shard*.log; do grep -q '"ran"' $f || { echo "  DID NOT FINISH $f"; grep -v pthread $f | tail -3 | cut -c1-240; }; done
done
rm -rf /tmp/aos-mp-$SLURM_JOB_ID
echo "end $(date -u +%FT%TZ)"
