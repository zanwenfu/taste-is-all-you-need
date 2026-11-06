#!/bin/bash
#SBATCH --job-name=aos-part-mem0
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=12:00:00
# Mem0's ingest, and its ranked lists from the question's words. No gold is read. FIRST and COUNT pick which of TOTAL shards this GPU runs, so several
# jobs can share the work and any can be run again: a unit that finished is skipped, one that did not is run whole. VIEW is lme or locomo. ARM names the
# rankings folder, EXTRA adds flags. With EMBEDDER set, that model is served on the same GPU and every process asks it. Free.
set -uo pipefail
source $STUDY_ROOT/code-${SHA:0:12}/studies/memory_bench/jobs/env.sh
VIEW=${VIEW:?lme or locomo}; TOTAL=${TOTAL:?}; FIRST=${FIRST:-0}; COUNT=${COUNT:-$TOTAL}; ARM=${ARM:-mem0}; EXTRA=${EXTRA:-}; EMBEDDER=${EMBEDDER:-}; DIMS=${DIMS:-384}
OUT=$R/rankings/$VIEW/$ARM; JOB=$OUT/job-$SLURM_JOB_ID
[ -s $V/$VIEW.jsonl ] || fail "no views file $V/$VIEW.jsonl: run the rank stage first"
mkdir -p $OUT $JOB
echo "start $(date -u +%FT%TZ) on $(hostname) code $TASTE_COMMIT part $PART $VIEW $ARM shards $FIRST+$COUNT of $TOTAL extra '$EXTRA' embedder '$EMBEDDER'"
gpu_log $JOB
SHARE=0.92; EMBED=(--embedder sentence-transformers/all-MiniLM-L6-v2)
if [ -n "$EMBEDDER" ]; then
  EPORT=$((20000 + RANDOM % 20000)); t0=$(date +%s)
  $RUN_GPU vllm serve $D/models/$EMBEDDER --served-model-name $EMBEDDER --runner pooling --host 127.0.0.1 --port $EPORT --gpu-memory-utilization 0.06 --max-model-len 8192 > $JOB/embedder.log 2>&1 &
  EMB=$!
  until curl -sf http://127.0.0.1:$EPORT/health >/dev/null; do
    sleep 5; kill -0 $EMB 2>/dev/null || { tail -30 $JOB/embedder.log; fail "embedder died"; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && fail "embedder load timeout"
  done
  echo "embedder load_seconds $(( $(date +%s) - t0 ))"
  SHARE=0.84; EMBED=(--embedder $EMBEDDER --embedder-url http://127.0.0.1:$EPORT/v1)
fi
serve_oss $JOB 65536 $SHARE
cd $CODE
t0=$(date +%s); SHARDS=()
for I in $(seq $FIRST $((FIRST+COUNT-1))); do
  $INSIDE $B/venv-mem0/bin/python -m studies.memory_bench.adapters.mem0_arm --views $V/$VIEW.jsonl --url http://127.0.0.1:$PORT/v1 --model $OSS_NAME \
    --max-tokens 4096 --effort low --date-prompt "${EMBED[@]}" --dims $DIMS --shard $I/$TOTAL --workdir /tmp/aos-mem0-$SLURM_JOB_ID-$I \
    --out $OUT/shard$I.jsonl $EXTRA > $OUT/shard$I.log 2>&1 &
  SHARDS+=($!)
done
wait "${SHARDS[@]}"
echo "shards ended after $(( $(date +%s) - t0 )) s"
FAILED=0
for I in $(seq $FIRST $((FIRST+COUNT-1))); do grep -q '"ran"' $OUT/shard$I.log || { FAILED=$((FAILED + 1)); echo "  SHARD $I DID NOT FINISH"; grep -v pthread $OUT/shard$I.log | tail -2 | cut -c1-200; }; done
metrics $JOB
if [ "$(ls $OUT/shard*[0-9].jsonl | wc -l)" = "$TOTAL" ]; then cat $OUT/shard*[0-9].jsonl > $OUT/rankings.jsonl; echo "$VIEW $ARM: $(wc -l < $OUT/rankings.jsonl) questions so far"; fi
$PY - $OUT $(( $(date +%s) - t0 )) <<'PY'
import collections, json, sys
from pathlib import Path
rows = [json.loads(l) for p in sorted(Path(sys.argv[1]).glob("shard*.replies.jsonl")) for l in p.read_text().splitlines()]
kinds = collections.Counter("parsed" if r["parsed"] else "no text" if r["reply"] == "null" else "cut" if not r["reply"].rstrip().endswith("}") else "other" for r in rows)
secs = sorted(r["seconds"] for r in rows)
print("  replies", len(rows), dict(kinds), "rate", kinds["parsed"] / max(1, len(rows)), "seconds median", secs[len(secs) // 2] if secs else None)
PY
kill $PID ${EMB:-} $SMI 2>/dev/null; wait $PID 2>/dev/null
rm -rf /tmp/aos-mem0-$SLURM_JOB_ID-*
echo "end $(date -u +%FT%TZ) shards failed $FAILED"
[ $FAILED = 0 ]
