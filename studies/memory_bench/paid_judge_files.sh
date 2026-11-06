#!/bin/bash
# The paid judge's files: the answers and the free judge's verdicts of the rows it reads, to the laptop, where the key is, and its verdicts back beside them on the cluster.
# No history and no key travel. Each file back is checked by md5. Needs HOST, and SSH_OPTS if the login needs them.
#
#   HOST=user@cluster bash paid_judge_files.sh fetch /path/on/cluster/runs/held_out-x copy/
#   HOST=user@cluster bash paid_judge_files.sh push copy/ /path/on/cluster/runs/held_out-x
set -euo pipefail
HOST=${HOST:?the login of the cluster}; SSH_OPTS=${SSH_OPTS:-}; NAME=answers.jsonl
ROWS=$(python3 -m studies.memory_bench.paid_judge rows)
case "${1:?fetch or push}" in
  fetch)
    RUN=${2:?}; COPY=${3:?}; mkdir -p "$COPY"
    FILES=$(for ROW in $ROWS; do echo "$ROW/$NAME $ROW/$NAME.eval-results-served"; done)
    ssh $SSH_OPTS "$HOST" "cd $RUN && tar cf - $(echo $FILES)" | tar xf - -C "$COPY"
    echo "fetched $(echo $ROWS | wc -w) rows"
    ;;
  push)
    COPY=${2:?}; RUN=${3:?}
    FILES=$(cd "$COPY" && for ROW in $ROWS; do ls $ROW/$NAME.eval-results-gpt-5.4 $ROW/$NAME.eval-results-gpt-5.4.settings.json 2>/dev/null || true; done)
    [ -n "$FILES" ] || { echo "no verdicts to send"; exit 1; }
    (cd "$COPY" && tar cf - $FILES) | ssh $SSH_OPTS "$HOST" "tar xf - -C $RUN"
    HERE=$(cd "$COPY" && md5sum $FILES 2>/dev/null || (cd "$COPY" && md5 -r $FILES))
    THERE=$(ssh $SSH_OPTS "$HOST" "cd $RUN && md5sum $(echo $FILES)")
    [ "$(echo "$HERE" | awk '{print $1}' | sort)" = "$(echo "$THERE" | awk '{print $1}' | sort)" ] || { echo "the copy differs"; exit 1; }
    echo "sent and checked $(echo $FILES | wc -w) files"
    ;;
esac
