#!/bin/bash
#SBATCH --job-name=aos-part-records
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
# The retrieval records, at every budget, from what each system ranked. This is the first stage that reads the gold, for retrieval scores, so on the held-out part
# it starts when the part opens, on the frozen commit. No model is called. SWEEP=0 leaves out the budget sweep. Free.
set -uo pipefail
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
SWEEP=${SWEEP:-1}; WORKERS=${SLURM_CPUS_PER_TASK:-8}; FAILED=0
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT part $PART"
cd $CODE
# What this commit makes of the benchmark's files must be what the rankings were ranked from, which may have come from an earlier commit.
mkdir -p /tmp/aos-r-$SLURM_JOB_ID
for row in "lme $LME_SPLIT $LME_DATA" "locomo $LOC_SPLIT $LOC_DATA"; do
  set -- $row
  $PY -m studies.memory_bench.run views --benchmark $1 --data $3 --split $2 --part $PART --out /tmp/aos-r-$SLURM_JOB_ID/views-$1.jsonl > /dev/null || fail "views $1"
  cmp -s /tmp/aos-r-$SLURM_JOB_ID/views-$1.jsonl $V/$1.jsonl || fail "the $1 views of this commit are not the ones the rankings came from"
done
one() {  # view system name budget tokenizer [extra]
  local view=$1 system=$2 name=$3 budget=$4 tok=$5; shift 5
  local split=$LME_SPLIT data=$LME_DATA
  [ $view = locomo ] && { split=$LOC_SPLIT; data=$LOC_DATA; }
  local out=$R/records/$view/$name-$budget; mkdir -p $out
  $PY -m studies.memory_bench.run retrieve --benchmark $view --data $data --split $split --part $PART --system $system --budget $budget --tokenizer $tok \
    --workdir /tmp/aos-r-$SLURM_JOB_ID --out $out/records.jsonl --workers $WORKERS ${FROZEN[@]+"${FROZEN[@]}"} "$@" > $out/run.log 2>&1 && echo "$view $name $budget $(tail -1 $out/run.log | cut -c1-120)" || { FAILED=$((FAILED + 1)); echo "FAILED $view $name $budget"; tail -3 $out/run.log | cut -c1-300; }
}
rival() { one $1 precomputed $2 $3 $LLAMA --rankings $R/rankings/$1/$2/rankings.jsonl; }
for VIEW in lme locomo; do cat $R/rankings/$VIEW/mem0/shard*[0-9].jsonl > $R/rankings/$VIEW/mem0/rankings.jsonl; done
for BUDGET in 8000 $( [ "$SWEEP" = 1 ] && echo 2000 4000 16000 ); do
  for VIEW in lme locomo; do
    for SYSTEM in flat_bm25 taste_git; do one $VIEW $SYSTEM $SYSTEM $BUDGET $LLAMA; done
    one $VIEW taste_git_headers taste_git_headers $BUDGET $LLAMA --headers $R/headers/$VIEW.jsonl
    [ $VIEW = lme ] && ARMS="flat_dense mem0 mempalace_bench_raw mempalace_bench_v4 mempalace_product" || ARMS="flat_dense mem0 mempalace_locomo_raw_session mempalace_locomo_raw_dialog mempalace_locomo_hybrid_session mempalace_product"
    for ARM in $ARMS; do rival $VIEW $ARM $BUDGET; done
  done
done
echo "--- the rows with no budget of their own"
for VIEW in lme locomo; do
  one $VIEW none none 8000 $LLAMA
  one $VIEW full newest 8000 $LLAMA
  one $VIEW full full_llama 31000 $LLAMA --charge prompt
  one $VIEW full full_oss 118000 $OSS --charge prompt
done
one lme oracle oracle_llama 31000 $LLAMA --charge prompt
one lme oracle oracle_oss 118000 $OSS --charge prompt
rm -rf /tmp/aos-r-$SLURM_JOB_ID
echo "--- the counts, and the equality the design says must hold"
$PY - $R <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
for view in ("lme", "locomo"):
    for folder in sorted((root / "records" / view).iterdir()):
        s = folder / "records.summary.json"
        if not s.exists():
            print("NO SUMMARY", view, folder.name); continue
        summary = json.loads(s.read_text())
        print(f"{view:7s} {folder.name:44s} n={summary['questions']:5d} tok={summary['tokens_delivered_mean']:9.0f} cut={summary['items_cut']:5d} dropped={summary['items_dropped_mean']:7.1f}")
    for budget in (2000, 4000, 8000, 16000):
        a, b = root / "records" / view / f"flat_bm25-{budget}" / "records.jsonl", root / "records" / view / f"taste_git-{budget}" / "records.jsonl"
        if a.exists() and b.exists():
            x, y = {r["qid"]: r for r in rows(a)}, {r["qid"]: r for r in rows(b)}
            print(view, budget, "keyword list and Taste with headers off: same ranking", sum(x[q]["ranked"] == y[q]["ranked"] for q in x), "same prompt", sum(x[q]["prompt_sha256"] == y[q]["prompt_sha256"] for q in x), "of", len(x))
PY
echo "end $(date -u +%FT%TZ) rows failed $FAILED"
[ $FAILED = 0 ]
