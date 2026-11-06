#!/bin/bash
# The stages of one part of the comparison, in the order they ran, each a committed job script. Run on the cluster's login node. It decides nothing itself.
#
#   STUDY_ROOT=... SHA=<commit> DIGEST=<its digest> PART=held_out bash submit_part.sh ingest
#   STUDY_ROOT=... SHA=<commit> DIGEST=<its digest> PART=held_out RUN=<first 12 of the ingest's commit> bash submit_part.sh open
#
# DIGEST is worked out from git where the repository is: python -c "from studies.memory_ledger import digest_of_commit; print(digest_of_commit('<commit>'))".
# ingest reads no gold: the shared files, MemPalace, the dense list, Mem0 across GPUs, the headers. open reads it, and on the held-out part the first stage waits for the part's
# date, which its own guard also checks. The reader's rows are split so that no job meets its time limit, the main rows first.
set -u
D=${STUDY_ROOT:?}; SHA=${SHA:?}; DIGEST=${DIGEST:?}; PART=${PART:?}; RUN=${RUN:-${SHA:0:12}}
J=$D/code-${SHA:0:12}/studies/memory_bench/jobs
L=$D/logs
OPENS=2026-11-10T12:00:00  # noon Eastern, the cluster's own zone
GPU="--partition=compsci-gpu --exclude=compsci-cluster-fitz-48"
CPU="--partition=compsci --exclude=compsci-cluster-fitz-48"
EXP="--export=ALL,STUDY_ROOT=$D,SHA=$SHA,DIGEST=$DIGEST,PART=$PART,RUN=$RUN"
ONE="--gres=gpu:rtx_pro_6000:1"
case "${1:?ingest or open}" in
  ingest)
    RANK=$(sbatch --parsable $CPU $EXP --output=$L/$PART-rank-%j.out $J/rank.sbatch)
    echo "rank $RANK"
    until [ -s $D/bench/runs/$PART-$RUN/views/locomo.jsonl ]; do sleep 10; done  # Mem0 and the headers need only the shared files
    for FIRST in 0 32 64 96; do
      sbatch --parsable $GPU $ONE --cpus-per-task=32 --mem=125G --time=06:00:00 $EXP,VIEW=lme,TOTAL=128,FIRST=$FIRST,COUNT=32 --output=$L/$PART-mem0-lme-$FIRST-%j.out $J/mem0.sbatch
    done
    sbatch --parsable $GPU $ONE --cpus-per-task=16 --mem=64G --time=04:00:00 $EXP,VIEW=locomo,TOTAL=8 --output=$L/$PART-mem0-locomo-%j.out $J/mem0.sbatch
    sbatch --parsable $GPU $ONE --time=06:00:00 $EXP --output=$L/$PART-headers-%j.out $J/headers.sbatch
    ;;
  open)
    BEGIN=(); [ "$PART" = held_out ] && BEGIN=(--begin=$OPENS)
    REC=$(sbatch --parsable $CPU $EXP --time=08:00:00 ${BEGIN[@]+"${BEGIN[@]}"} --output=$L/$PART-records-%j.out $J/records.sbatch)
    AFTER="--dependency=afterok:$REC"
    MAIN='-8000 |-118000 '
    R1=$(sbatch --parsable $GPU $ONE $EXP,READER=oss,ONLY="lme/.*($MAIN)" $AFTER --output=$L/$PART-read-oss-lme-%j.out $J/read.sbatch)
    R2=$(sbatch --parsable $GPU $ONE $EXP,READER=oss,ONLY="locomo/.*($MAIN)" $AFTER --output=$L/$PART-read-oss-locomo-%j.out $J/read.sbatch)
    R3=$(sbatch --parsable $GPU $ONE $EXP,READER=oss,ONLY='lme/.*-(2000|4000|16000) ' $AFTER --output=$L/$PART-read-oss-lme-sweep-%j.out $J/read.sbatch)
    R4=$(sbatch --parsable $GPU $ONE $EXP,READER=oss,ONLY='locomo/.*-(2000|4000|16000) ' $AFTER --output=$L/$PART-read-oss-locomo-sweep-%j.out $J/read.sbatch)
    R5=$(sbatch --parsable $GPU --gres=gpu:rtx_pro_6000:2 --exclude=compsci-cluster-fitz-47,compsci-cluster-fitz-48 $EXP,READER=llama --time=08:00:00 $AFTER --output=$L/$PART-read-llama-%j.out $J/read.sbatch)
    JU=$(sbatch --parsable $GPU $ONE $EXP --dependency=afterok:$R1:$R2:$R3:$R4:$R5 --output=$L/$PART-judge-%j.out $J/judge.sbatch)
    echo "records $REC read $R1 $R2 $R3 $R4 $R5 judge $JU"
    ;;
esac
