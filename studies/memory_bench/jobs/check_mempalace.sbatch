#!/bin/bash
#SBATCH --job-name=aos-check-mempalace
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Does the adapter rank LoCoMo as MemPalace's own run did, given the same words? All ten conversations with captions left out, as its runner leaves them, raw and hybrid at
# session level, against the lists it committed at its release. No model and no gold: the recall printed is the one in its own file. It committed no list at dialog level, so
# that arm is held to its runner's number instead, by the job that ran its runner. Before the held-out part opens. Free, CPU.
set -uo pipefail
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
OUT=$B/runs/check-mempalace-$S; mkdir -p $OUT
export MEMPALACE_CONFIG_DIR=$OUT/mempalace-config MEMPALACE_EMBEDDING_MODEL=minilm MEMPALACE_EMBEDDING_DEVICE=cpu MEMPALACE_EMBEDDING_THREADS=2
mkdir -p $MEMPALACE_CONFIG_DIR
FAILED=0
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT"
cd $CODE
$PY -m studies.memory_bench.run views --benchmark locomo --no-captions --data $LOC_DATA --split $LOC_SPLIT --part all --out $OUT/views.jsonl | cut -c1-200 || fail "views"
for PAIR in locomo_raw_session:results_locomo_raw_session_top10_20260414_1634.json locomo_hybrid_session:results_locomo_hybrid_session_top10_20260414_1649.json; do
  ARM=${PAIR%%:*}; FILE=${PAIR#*:}; T0=$(date +%s); PIDS=()
  for I in $(seq 0 9); do
    $INSIDE $B/venv-mempalace/bin/python -m studies.memory_bench.adapters.mempalace_arms --views $OUT/views.jsonl --repo $B/src/mempalace \
      --arm $ARM --workdir /tmp/aos-cm-$SLURM_JOB_ID --shard $I/10 --out $OUT/$ARM-shard$I.jsonl > $OUT/$ARM-shard$I.log 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  cat $OUT/$ARM-shard?.jsonl > $OUT/$ARM.jsonl
  echo "$ARM: $(wc -l < $OUT/$ARM.jsonl) questions, $(( $(date +%s) - T0 )) s"
  $PY -m studies.memory_bench.adapters.check_mempalace --locomo --published $B/src/mempalace/benchmarks/$FILE --rankings $OUT/$ARM.jsonl --least-same 0.995 > $OUT/check-$ARM.json
  code=$?; [ $code = 0 ] || FAILED=$((FAILED + 1))
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ('compared','same_top10','published_mean_recall')}, 'differing', len(d['differing_top10']))" $OUT/check-$ARM.json
done
rm -rf /tmp/aos-cm-$SLURM_JOB_ID
echo "end $(date -u +%FT%TZ) failed $FAILED"
[ $FAILED = 0 ]
