"""LongMemEval's judge, run on Amazon Bedrock instead of OpenAI.

The prompts are LongMemEval's own, byte for byte (src/evaluation/evaluate_qa.py at
commit 9e0b455, lines 24-43), and so are the routing (abstention by an ``_abs`` in the
question id, then the question type) and the output file
(``<hypotheses>.eval-results-<judge>``, one JSON line per hypothesis with
``autoeval_label``).

What differs from the paper, and why:
- The judge. The paper's is ``gpt-4o-2024-08-06`` through OpenAI, which this study
  cannot use. Two Bedrock models stand in: GPT-5.4 and gpt-oss-120b. Both reason before
  they answer, so the paper's 10-token cap would cut them off; the cap here is set by
  ``Converse.max_tokens`` and recorded with each verdict.
- The parse. The paper counts a reply as correct if "yes" appears anywhere in it,
  which reads "No, yesterday..." as yes. Here a verdict must be one clear "yes" or "no";
  anything else is recorded as unparseable, kept in the file and counted, never
  silently dropped. The paper's own rule is kept beside it (``official_rule_label``)
  so both can be reported.
- Temperature is not sent (see bedrock.py); the paper used 0.

    python -m studies.context_rot.lme_judge <hypotheses.jsonl> <reference.json> --judge gpt-5.4
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studies.context_rot.bedrock import Converse

TYPES = ("single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning",
         "knowledge-update", "single-session-preference")
JUDGES = {"gpt-5.4": "global.openai.gpt-5.4", "gpt-oss-120b": "openai.gpt-oss-120b-1:0"}

_ABSTENTION = 'I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.'
_PLAIN = 'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.'
_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_UPDATE = 'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.'
_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."


def anscheck_prompt(task: str, question, answer, response, abstention: bool = False) -> str:
    """The prompt LongMemEval's ``get_anscheck_prompt`` builds for these arguments."""
    if abstention:
        return _ABSTENTION.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return _PLAIN.format(question, answer, response)
    if task == "temporal-reasoning":
        return _TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return _UPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return _PREFERENCE.format(question, answer, response)
    raise NotImplementedError(task)


_WRAPPING = " \t\n\r\"'`*_.!:"


def parse_verdict(text: str) -> bool | None:
    """True for one clear "yes", False for one clear "no", None for anything else."""
    word = (text or "").strip().strip(_WRAPPING).strip().lower()
    return {"yes": True, "no": False}.get(word)


def official_label(text: str) -> bool:
    """The paper's rule: correct if "yes" appears anywhere (evaluate_qa.py:113)."""
    return "yes" in (text or "").lower()


def _load(path: Path) -> list[dict]:
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return value if isinstance(value, list) else [value]


class SpendLimit(RuntimeError):
    """The spend limit was reached. Nothing more was asked, and what was judged is kept."""


def _spent(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(json.loads(line)["autoeval_label"].get("cost_usd_estimate") or 0.0 for line in path.read_text().splitlines())


def judge_file(hyp_path, ref_path, judge: Converse, *, short_name: str, workers: int = 8, stop_at_usd: float | None = None,
               spent_elsewhere: float = 0.0) -> Path:
    """Judge every hypothesis not yet in the output file; return the output file. With ``stop_at_usd``, no call starts once this file's
    spend and ``spent_elsewhere`` reach it. Calls already out may finish, so the limit can pass by what ``workers`` calls cost."""
    hyp_path, ref_path = Path(hyp_path), Path(ref_path)
    out = hyp_path.with_name(f"{hyp_path.name}.eval-results-{short_name}")
    done = {json.loads(line)["question_id"] for line in out.read_text().splitlines()} if out.exists() else set()
    spent = [_spent(out) + spent_elsewhere]
    refs = {entry["question_id"]: entry for entry in _load(ref_path)}
    todo = [entry for entry in _load(hyp_path) if entry["question_id"] in refs and entry["question_id"] not in done]
    skipped = [entry["question_id"] for entry in _load(hyp_path) if entry["question_id"] not in refs]
    if skipped:
        print(f"warning: {len(skipped)} hypotheses are not in the reference data", file=sys.stderr)
    lock = threading.Lock()

    def one(entry: dict) -> None:
        with lock:
            if stop_at_usd is not None and spent[0] >= stop_at_usd:
                return
        ref = refs[entry["question_id"]]
        prompt = anscheck_prompt(ref["question_type"], ref["question"], ref["answer"], entry["hypothesis"],
                                 abstention="_abs" in entry["question_id"])
        reply = judge.ask(prompt)
        label = parse_verdict(reply.text)
        entry = {**entry, "autoeval_label": {
            "model": judge.model, "label": label,
            "parse": "unparseable" if label is None else ("yes" if label else "no"),
            "official_rule_label": official_label(reply.text), "judge_output": reply.text,
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            "stop_reason": reply.stop_reason, "effort": judge.effort, "max_tokens": judge.max_tokens,
            "cost_usd_estimate": reply.cost(judge.model)}}
        with lock, out.open("a") as written:
            written.write(json.dumps(entry) + "\n")
            spent[0] += reply.cost(judge.model) or 0.0

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    if not out.exists():
        out.write_text("")
    if stop_at_usd is not None and spent[0] >= stop_at_usd:
        unjudged = len(todo) + len(done) - len(out.read_text().splitlines())
        if unjudged:
            raise SpendLimit(f"stopped at ${spent[0]:.4f} of a ${stop_at_usd:.4f} limit, with {unjudged} answers not judged")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hypotheses")
    parser.add_argument("reference")
    parser.add_argument("--judge", choices=sorted(JUDGES), required=True)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stop-at-usd", type=float, help="start no call once this much has been spent, by the prices in bedrock.py")
    parser.add_argument("--spent-elsewhere", type=float, default=0.0, help="what the same job already spent on other files")
    arguments = parser.parse_args(argv)
    judge = Converse(JUDGES[arguments.judge], effort=arguments.effort, max_tokens=arguments.max_tokens)
    print(judge_file(arguments.hypotheses, arguments.reference, judge, short_name=arguments.judge, workers=arguments.workers,
                     stop_at_usd=arguments.stop_at_usd, spent_elsewhere=arguments.spent_elsewhere))
    return 0


if __name__ == "__main__":
    sys.exit(main())
