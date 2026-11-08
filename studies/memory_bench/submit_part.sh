#!/bin/bash
# The stages of one part of the comparison, in the order they ran, each a committed job script. Run on the cluster's login node. It decides nothing itself.
#
#   STUDY_ROOT=... SHA=<commit> DIGEST=<its digest> PART=held_out bash submit_part.sh ingest
#   STUDY_ROOT=... SHA=<commit> DIGEST=<its digest> PART=held_out RUN=<first 12 of the ingest's commit> bash submit_part.sh open
#
# DIGEST is worked out from git where the repository is: python -c "from studies.memory_ledger import digest_of_commit; print(digest_of_commit('<commit>'))".
# ingest reads no gold: the shared files, MemPalace, the dense list, Mem0 across GPUs, the headers. open reads it, and on the held-out part the first stage waits for the part's
# date, which its own guard also checks. The reader's rows are split so that no job meets its time limit, the main rows first.
#
#   STUDY_ROOT=... SHA=<commit> DIGEST=<its digest> PART=held_out RUN=<...> OSS_JOBS="<job> <job>" LLAMA_JOBS="<job> <job>" bash submit_part.sh steps
#
# steps does what open does on GPUs the study already has, for when the queue would keep the stages from their day: the same job scripts and the same rows, each a task on a job running
# long_run/jobs/keep.sbatch with a server task, read with SERVER_URL. It hands over the tasks and exits, and the tasks keep the order themselves (jobs/stage.sh), so that nothing rests on a login
# node. Run again, with AGAIN set to name the new tasks, it asks again only for stages with no mark of having ended well, and every row asks only what has no answer.
set -u
D=${STUDY_ROOT:?}; SHA=${SHA:?}; DIGEST=${DIGEST:?}; PART=${PART:?}; RUN=${RUN:-${SHA:0:12}}
J=$D/code-${SHA:0:12}/studies/memory_bench/jobs
L=$D/logs
OPENS=2026-11-08T08:00:00  # 8 am Eastern, the cluster's own zone, as in run.py (moved from noon Tue Nov 10 on Nov 8, by the user)
GPU="--partition=compsci-gpu --exclude=compsci-cluster-fitz-48"
CPU="--partition=compsci --exclude=compsci-cluster-fitz-48"
EXP="--export=ALL,STUDY_ROOT=$D,SHA=$SHA,DIGEST=$DIGEST,PART=$PART,RUN=$RUN"
ONE="--gres=gpu:rtx_pro_6000:1"
case "${1:?ingest, open or steps}" in
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
    R5=$(sbatch --parsable $GPU --gres=gpu:rtx_pro_6000:2 --exclude=compsci-cluster-fitz-47,compsci-cluster-fitz-48 $EXP,READER=llama --time=30:00:00 $AFTER --output=$L/$PART-read-llama-%j.out $J/read.sbatch)
    # The first judge does not wait on Llama, whose two GPUs may come late; the second follows it, since two judges must not write one verdict file at once, and asks only what has no verdict.
    JU=$(sbatch --parsable $GPU $ONE $EXP --dependency=afterok:$R1:$R2:$R3:$R4 --output=$L/$PART-judge-%j.out $J/judge.sbatch)
    JL=$(sbatch --parsable $GPU $ONE $EXP --dependency=afterok:$R5,afterany:$JU --output=$L/$PART-judge-llama-%j.out $J/judge.sbatch)
    echo "records $REC read $R1 $R2 $R3 $R4 $R5 judge $JU $JL"
    ;;
  steps)
    read -r -a OSS <<< "${OSS_JOBS:?the kept jobs that run gpt-oss}"; read -r -a LLAMA <<< "${LLAMA_JOBS:?the kept jobs that run Llama}"; AGAIN=${AGAIN:-}; LAST=$(( ${#OSS[@]} - 1 )); LAST_LLAMA=$(( ${#LLAMA[@]} - 1 ))
    # Llama's rows the comparison turns on, which go first on each server; and with two servers, the share of LoCoMo the first takes once LongMemEval is read, so that neither waits
    # long on the other, by LoCoMo's own row names (LLAMA_SHARE at hand-over to name others). The share leaves the comparison's rows alone, as the other server reads them first: two
    # stages on one row would write one answers file at once. By the rehearsal's times, scaled to the held-out part (LongMemEval 9 times dev, LoCoMo 1,551 of 435 questions): the
    # first reads LongMemEval in about 7.7 hours and then dense and MemPalace raw session in 2.8; the second the comparison's five LoCoMo rows in 7.4 and the other five in 3.1.
    FIRST='/(taste_git|taste_git_headers|flat_bm25|mem0|mempalace_product)-'
    SHARE=${LLAMA_SHARE:-/(flat_dense|mempalace_locomo_raw_session)-}
    if [ $LAST_LLAMA -gt 0 ]; then
      NAMES=${FIRST#/(}; NAMES=${NAMES%)-}
      for NAME in ${NAMES//|/ }; do
        if echo "locomo/$NAME-8000 " | grep -qE "locomo$SHARE"; then echo "the first server's share would take $NAME, which the comparison turns on: name another share"; exit 1; fi
      done
    fi
    KEEP=$D/code-${SHA:0:12}/studies/long_run/jobs; MARKS=$D/bench/runs/$PART-$RUN/stages; mkdir -p "$MARKS"
    if compgen -G "$MARKS/*.failed" > /dev/null; then
      [ -n "$AGAIN" ] || { echo "stages have failed: AGAIN must name the new tasks"; exit 1; }
      for FAILED in "$MARKS"/*.failed; do mv "$FAILED" "$FAILED.before$AGAIN"; done
    fi
    GATE=; [ "$PART" = held_out ] && GATE=$(date -d $OPENS +%s)
    # A kept job with its GPUs at rest ends itself after 8 hours, whatever is waiting on it.
    if [ -n "$GATE" ] && [ $(( GATE - $(date +%s) )) -gt 21600 ]; then echo "more than 6 hours before the hour: hand these over nearer it"; exit 1; fi
    quiet() {  # name: seconds since its stage last said it was alive, or nothing from a stage on code from before the beat
      local at; at=$(cat "$MARKS/$1.beat" 2>/dev/null)
      case "$at" in ''|*[!0-9]*) return 0;; esac
      echo $(( $(date +%s) - at ))
    }
    stage() {  # name job server script after [more of the task's settings]
      if [ "$(cat "$MARKS/$1.ok" 2>/dev/null)" = "$SHA" ]; then echo "$1 has ended well"; return 0; fi
      if [ -e "$MARKS/$1.running" ] && ! bash "$KEEP/job_gone.sh" "$(cat "$MARKS/$1.running")"; then
        QUIET=$(quiet "$1")
        # Silent on a job still up, its shell may yet be alive behind a stalled file system, and a second copy would write the same rows: a person looks first.
        if [ -n "$QUIET" ] && [ "$QUIET" -ge "${STAGE_SILENT_SECONDS:-1800}" ]; then
          echo "$1 has said nothing for $QUIET seconds on job $(cat "$MARKS/$1.running"): not asked again, as it may still be alive; look at its task"; return 0
        fi
        echo "$1 is still running on job $(cat "$MARKS/$1.running"): not asked again"; return 0
      fi
      # What a job that has gone left behind is set aside, so that what follows does not take the new task, not yet started, for the one that was lost.
      for LEFT in running beat; do [ -e "$MARKS/$1.$LEFT" ] && mv "$MARKS/$1.$LEFT" "$MARKS/$1.$LEFT.before$AGAIN"; done
      printf 'export STUDY_ROOT=%q SHA=%q DIGEST=%q PART=%q RUN=%q OPENS_EPOCH=%q SERVER_WAIT=%q %s\nexec bash %q %q %q %q %q\n' "$D" "$SHA" "$DIGEST" "$PART" "$RUN" "$GATE" "${SERVER_WAIT:-2400}" "${6:-}" "$J/stage.sh" "$1" "$3" "$4" "$5" |
        STUDY_ROOT=$D bash "$KEEP/give_task.sh" "$2" "$PART-$1$AGAIN"
    }
    MAIN='-8000 |-118000 '
    stage records "${OSS[0]}" none records.sbatch ""
    stage read-oss-lme "${OSS[0]}" gpt-oss read.sbatch records "READER=oss ONLY='lme/.*($MAIN)'"
    stage read-oss-locomo "${OSS[0]}" gpt-oss read.sbatch records "READER=oss ONLY='locomo/.*($MAIN)'"
    stage read-oss-lme-sweep "${OSS[$LAST]}" gpt-oss read.sbatch records "READER=oss ONLY='lme/.*-(2000|4000|16000) '"
    stage read-oss-locomo-sweep "${OSS[$LAST]}" gpt-oss read.sbatch records "READER=oss ONLY='locomo/.*-(2000|4000|16000) '"
    # Llama is the slow reader: on the rehearsal LongMemEval took 51 minutes and LoCoMo 3.7 hours, so the held-out part is about 21 hours on one server. A second server takes LoCoMo
    # but for the first server's share. On one server LongMemEval goes first, as the second judge waits only for that. On each, the comparison's rows go first.
    THEN=; [ $LAST_LLAMA = 0 ] && THEN=" read-llama-lme?"
    stage read-llama-lme-first "${LLAMA[0]}" llama read.sbatch records "READER=llama ONLY='lme$FIRST'"
    stage read-llama-lme "${LLAMA[0]}" llama read.sbatch "records read-llama-lme-first?" "READER=llama ONLY='lme/'"
    stage read-llama-locomo-first "${LLAMA[$LAST_LLAMA]}" llama read.sbatch "records$THEN" "READER=llama ONLY='locomo$FIRST'"
    if [ $LAST_LLAMA = 0 ]; then
      stage read-llama-locomo "${LLAMA[0]}" llama read.sbatch "records read-llama-locomo-first?" "READER=llama ONLY='locomo/'"
    else
      stage read-llama-locomo "${LLAMA[$LAST_LLAMA]}" llama read.sbatch "records read-llama-locomo-first?" "READER=llama ONLY='locomo/' NOT='locomo$SHARE'"
      stage read-llama-locomo-share "${LLAMA[0]}" llama read.sbatch "records read-llama-lme?" "READER=llama ONLY='locomo$SHARE'"
    fi
    # The first judge does not wait on Llama, and two judges must not write one verdict file at once: the second follows it, however it ended. Only LongMemEval's answers are judged.
    stage judge "${OSS[0]}" gpt-oss judge.sbatch "read-oss-lme read-oss-locomo read-oss-lme-sweep read-oss-locomo-sweep"
    stage judge-llama "${OSS[0]}" gpt-oss judge.sbatch "read-llama-lme judge?"
    ;;
esac
