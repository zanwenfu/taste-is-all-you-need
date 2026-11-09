"""Calendar dates written into a raw history at write time, with no model call (Nov 9).

Mem0, as the comparison runs it, is given each session's date when it stores a memory, and its write-time model turns
"yesterday" into a calendar date inside the fact it keeps. A raw history keeps "yesterday" and shows the session's date
only in a header line, so a reader must do the arithmetic itself. This pass gives the raw history the same help without
a model: each relative expression this module knows is followed by the date it names, worked out from the session's
date, in brackets. The words themselves are kept as they were. Expressions that could name two dates ("on Friday",
"this Friday", "a few days ago") are left alone.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable

_MONTHS = {name: number for number, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"), 1)}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "a couple of": 2, "couple of": 2}
_COUNT = r"(?P<count>\d{1,3}|a couple of|couple of|an|a|one|two|three|four|five|six|seven|eight|nine|ten)"
_NOT_DONE = r"(?! \[)"  # an expression already followed by its date is not dated again


def session_day(date: str) -> dt.date | None:
    """The calendar day of a session's date as each benchmark writes it: LongMemEval "2023/05/20 (Sat) 02:21",
    LoCoMo "1:56 pm on 8 May, 2023". Anything else gives None, and then nothing is written."""
    found = re.search(r"(\d{4})/(\d{2})/(\d{2})", date)
    if found:
        return dt.date(int(found[1]), int(found[2]), int(found[3]))
    found = re.search(r"(\d{1,2}) (" + "|".join(_MONTHS) + r"),? (\d{4})", date, re.IGNORECASE)
    if found:
        return dt.date(int(found[3]), _MONTHS[found[2].lower()], int(found[1]))
    return None


def _months(day: dt.date, months: int) -> str:
    index = day.year * 12 + day.month - 1 + months
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _count(match: re.Match) -> int:
    word = match["count"].lower()
    return int(word) if word.isdigit() else _COUNTS[word]


def _shifted(day: dt.date, unit: str, amount: int) -> str:
    unit = unit.lower()
    if unit == "day":
        return (day + dt.timedelta(days=amount)).isoformat()
    if unit == "week":
        return (day + dt.timedelta(weeks=amount)).isoformat()
    if unit == "month":
        return _months(day, amount)
    return f"{day.year + amount:04d}"


def _rules(day: dt.date) -> list[tuple[str, Callable[[re.Match], str]]]:
    monday = day - dt.timedelta(days=day.weekday())
    week = lambda shift: f"the week of {(monday + dt.timedelta(weeks=shift)).isoformat()}"  # noqa: E731
    plain = lambda shift: lambda match: (day + dt.timedelta(days=shift)).isoformat()  # noqa: E731

    def weekday(match: re.Match) -> str:
        wanted = _WEEKDAYS.index(match["day"].lower())
        if match["which"].lower() == "last":
            return (day - dt.timedelta(days=(day.weekday() - wanted) % 7 or 7)).isoformat()
        return (day + dt.timedelta(days=(wanted - day.weekday()) % 7 or 7)).isoformat()

    shifts = {"last": -1, "this": 0, "next": 1}
    return [
        (r"\bthe day before yesterday\b", plain(-2)),
        (r"\bthe day after tomorrow\b", plain(2)),
        (r"\b(?:earlier today|later today|this morning|this afternoon|this evening|tonight|today)\b", plain(0)),
        (r"\byesterday\b", plain(-1)),
        (r"\btomorrow\b", plain(1)),
        (_COUNT + r" (?P<unit>day|week|month|year)s? ago\b", lambda match: _shifted(day, match["unit"], -_count(match))),
        (r"\bin " + _COUNT + r" (?P<unit>day|week|month|year)s?\b", lambda match: _shifted(day, match["unit"], _count(match))),
        (r"\b(?P<which>last|this) weekend\b",
         lambda match: f"the weekend of {(monday + dt.timedelta(days=5 if match['which'].lower() == 'this' else -2)).isoformat()}"),
        (r"\b(?P<which>last|this|next) week\b", lambda match: week(shifts[match["which"].lower()])),
        (r"\b(?P<which>last|this|next) month\b", lambda match: _months(day, shifts[match["which"].lower()])),
        (r"\b(?P<which>last|this|next) year\b", lambda match: f"{day.year + shifts[match['which'].lower()]:04d}"),
        (r"\b(?P<which>last|next) (?P<day>" + "|".join(_WEEKDAYS) + r")\b", weekday),
    ]


def write_dates(text: str, day: dt.date) -> str:
    """The text with each relative expression this module knows followed by its date in brackets."""
    for pattern, named in _rules(day):
        text = re.sub(pattern + _NOT_DONE, lambda match: f"{match[0]} [{named(match)}]", text, flags=re.IGNORECASE)
    return text


def dated_session(session: dict) -> dict:
    """A session with its turns' text dated from its own date; one whose date cannot be read is kept as it was."""
    day = session_day(str(session.get("date", "")))
    if day is None:
        return session
    turns = [{**turn, "content": write_dates(turn["content"], day)} if isinstance(turn.get("content"), str) else turn
             for turn in session["turns"]]
    return {**session, "turns": turns}
