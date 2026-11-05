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
from pathlib import Path

from studies.context_rot.lme_clean import neutral


def compare(data: list[dict], published: list[dict], rankings: list[dict]) -> dict:
    by_question = {item["question_id"]: item for item in data}
    ours = {row["qid"]: [i["session"] for i in row["items"]] for row in rankings}
    same_top10 = same_top5_set = compared = 0
    recall = {"published@5": 0, "published@10": 0, "adapter@5": 0, "adapter@10": 0}
    differing = []
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
    return {"compared": compared, "same_top10": same_top10, "same_top5_set": same_top5_set,
            "recall_any": {key: value / compared for key, value in recall.items()} if compared else {},
            "differing_top5": differing[:50]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("data", "published", "rankings"):
        parser.add_argument(f"--{name}", required=True)
    arguments = parser.parse_args(argv)
    data = json.loads(Path(arguments.data).read_text())
    published = [json.loads(line) for line in Path(arguments.published).read_text().splitlines() if line.strip()]
    rankings = [json.loads(line) for line in Path(arguments.rankings).read_text().splitlines() if line.strip()]
    print(json.dumps(compare(data, published, rankings), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
