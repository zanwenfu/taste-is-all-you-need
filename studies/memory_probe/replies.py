"""What a writer's reply holds: a summary by what its prompt asked for, or work carried on in place of writing.

Moved here unchanged from the long run's session (where these were found on Laguna S 2.1, Nov 6 2026), so that the matrix's
writers can read a second model's replies by the same rule without the long run's agent and its packages.
"""

from __future__ import annotations

import re

_SUMMARY_BLOCK = re.compile(r"(?:<summary>|\[summary\])(.*?)(?:</summary>|\[/summary\])", re.S)
_SUMMARY_OPEN = re.compile(r"^\s*(?:<summary>|\[summary\])(.*)$", re.S)
_WORK_MARK = re.compile(r"</?tool_call>|</?arg_key>|</?arg_value>|</think>")
_WORK_LINE = re.compile(r"^\s*(?:\[bash\]\s*\{|<bash\b)", re.M)  # a line that is a call, in the transcript's own form; a summary may name one mid-sentence
_PI_FIRST, _PI_LAST = re.compile(r"^## Goal\b", re.M), re.compile(r"^## Next Steps\b", re.M)
_FEWEST_WORDS = 5  # a mention of the form, such as <summary>...</summary>, is not a summary


def carries_on_the_work(text: str) -> re.Match | None:
    """Where a reply turns from writing into working, if it does."""
    found = [match for match in (_WORK_MARK.search(text), _WORK_LINE.search(text)) if match]
    return min(found, key=lambda match: match.start()) if found else None


def a_summary(text: str) -> bool:
    return len(re.findall(r"\w+", text)) >= _FEWEST_WORDS


def read_summary(option: str, reply: str) -> str | None:
    """The summary in a reply, by what its prompt asked for, or None if there is none: Anthropic's asks for a block, whose closing tag some models leave out, and Pi's for its headings, each on a
    line of its own. What comes before the end of the model's thinking is thinking, whatever it mentions."""
    text = reply.rsplit("</think>", 1)[-1]
    if option == "pi_mono":
        first = _PI_FIRST.search(text)
        if not first:
            return None
        text = text[first.start():]
        work = carries_on_the_work(text)
        text = (text[:work.start()] if work else text).strip()
        return text if _PI_LAST.search(text) and a_summary(text) else None
    blocks = [block.strip() for block in _SUMMARY_BLOCK.findall(text)]
    opened = _SUMMARY_OPEN.match(text)
    if not blocks and opened:
        blocks = [opened.group(1).strip()]
    whole = [block for block in blocks if a_summary(block) and not carries_on_the_work(block)]
    return max(whole, key=len) if whole else None
