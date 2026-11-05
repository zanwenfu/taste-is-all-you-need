"""Does the MemPalace adapter rank each LongMemEval question as MemPalace's own published run did?

MemPalace's repository commits, at the release, the per-question ranked lists behind its headline numbers. This
maps their session ids to the study's neutral ids, compares each question's top ten with the adapter's, and gives
recall at 5 and 10 from both, against the benchmark's answer sessions. Near-ties may flip with another ONNX or
Chroma build; a systematic gap means the adapter is wrong.

    python -m studies.memory_bench.adapters.check_mempalace --data S.json --published results.jsonl \\
        --rankings rankings.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from studies.context_rot.lme_clean import neutral


def compare(data: list[dict], published: list[dict], rankings: list[dict]) -> dict:
    by_question = {item["question_id"]: item for item in data}
    ours = {row["qid"]: [i["session"] for i in row["items"]] for row in rankings}
    same_top10 = same_top5_set = compared = 0
    recall = {"published@5": 0, "published@10": 0, "adapter@5": 0, "adapter@10": 0}
    differing, differing_lists = [], []
    for row in published:
        question = row["question_id"]
        qid = neutral("q", question)
        if qid not in ours:
            continue
        compared += 1
        theirs = list(dict.fromkeys(neutral("s", f"{question}/{item['corpus_id']}")
                                    for item in row["retrieval_results"]["ranked_items"]))
        mine = list(dict.fromkeys(ours[qid]))
        gold = {neutral("s", f"{question}/{sid}") for sid in by_question[question]["answer_session_ids"]}
        same_top10 += theirs[:10] == mine[:10]
        same_top5_set += set(theirs[:5]) == set(mine[:5])
        for name, ranked in (("published", theirs), ("adapter", mine)):
            for k in (5, 10):
                recall[f"{name}@{k}"] += bool(gold & set(ranked[:k]))
        if theirs[:5] != mine[:5]:
            differing.append(question)
        if theirs[:10] != mine[:10]:
            first = next(rank for rank, pair in enumerate(zip(theirs, mine, strict=False), 1) if pair[0] != pair[1])
            differing_lists.append({"question": question, "first_rank": first,
                                    "published": theirs[:10], "adapter": mine[:10]})
    return {"compared": compared, "same_top10": same_top10, "same_top5_set": same_top5_set,
            "recall_any": {key: value / compared for key, value in recall.items()} if compared else {},
            "differing_top5": differing[:50], "differing_top10": differing_lists[:50]}


def compare_locomo(published: list[dict], rankings: list[dict]) -> dict:
    """Its LoCoMo runner's committed lists, in the dataset's own question order, against a session-level arm's. The
    runner gives it the words of each turn; the study's ingest adds an image's caption, so a few near-ties may flip."""
    ours = {row["qid"]: list(dict.fromkeys(item["session"] for item in row["items"])) for row in rankings}
    position: Counter = Counter()
    compared = same = 0
    recall, differing = [], []
    for row in published:
        sample = row["sample_id"]
        qid = neutral("q", f"{sample}:{position[sample]}")
        position[sample] += 1
        if qid not in ours:
            continue
        theirs = [neutral("s", f"{sample}/{corpus_id.split('_')[1]}") for corpus_id in row["retrieved_ids"]]
        compared += 1
        recall.append(row["recall"])
        if ours[qid][: len(theirs)] == theirs:
            same += 1
        else:
            differing.append(qid)
    return {"compared": compared, "same_top10": same, "published_mean_recall": sum(recall) / compared if compared else None,
            "differing_top10": differing[:50]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("published", "rankings"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--data", help="LongMemEval's file; not needed with --locomo")
    parser.add_argument("--locomo", action="store_true", help="compare with its committed LoCoMo lists")
    arguments = parser.parse_args(argv)
    rankings = [json.loads(line) for line in Path(arguments.rankings).read_text().splitlines() if line.strip()]
    if arguments.locomo:
        print(json.dumps(compare_locomo(json.loads(Path(arguments.published).read_text()), rankings), indent=1))
        return 0
    data = json.loads(Path(arguments.data).read_text())
    published = [json.loads(line) for line in Path(arguments.published).read_text().splitlines() if line.strip()]
    print(json.dumps(compare(data, published, rankings), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
