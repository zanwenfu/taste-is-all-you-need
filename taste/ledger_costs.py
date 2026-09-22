"""Validated lifetime costs shared by sweep admission and offline reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def lifetime_billed_usd(row: Mapping[str, Any]) -> float:
    """Sum every recorded attempt; unavailable historical costs are errors.

    Old single-attempt rows may omit attempt metadata. A row identifying
    multiple attempts must contain the complete flat history, even when its
    latest attempt was free. A missing cost is never replaced by a cap or zero.
    """
    attempts = row.get("attempts_made", 1)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("cell attempt count must be a positive integer")
    prior = row.get("prior_attempts", ())
    if not isinstance(prior, (tuple, list)) or len(prior) != attempts - 1:
        raise ValueError("cell is missing earlier attempt costs; refusing to assume zero")
    identity = tuple(row.get(key) for key in ("task", "arm", "trial"))
    costs = [row.get("billed_usd")]
    for number, item in enumerate(prior, 1):
        if (not isinstance(item, dict) or isinstance(item.get("attempts_made"), bool)
                or item.get("attempts_made") != number
                or tuple(item.get(key) for key in ("task", "arm", "trial")) != identity
                or item.get("prior_attempts")):
            raise ValueError("cell attempt history has inconsistent identity or is not flat")
        costs.append(item.get("billed_usd"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("cell attempt costs must be finite and non-negative")
    try:
        total = math.fsum(costs)
    except OverflowError as exc:
        raise ValueError("cell lifetime cost overflowed") from exc
    if not math.isfinite(total):
        raise ValueError("cell lifetime cost must be finite")
    return total


def read_cost_rows(directory: Path) -> list[dict[str, Any]]:
    """Read the cell ledger without creating it or silently dropping bad rows."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"cell ledger directory is missing: {directory}")
    if (directory / ".sweep-journal" / "pending.json").exists():
        raise ValueError("a sweep attempt is pending; final lifetime spending is not settled")
    rows = []
    for path in sorted(directory.glob("*.json")):
        try:
            row = json.loads(path.read_text())
            if not isinstance(row, dict):
                raise ValueError("cell result must be an object")
            task, arm, trial = row.get("task"), row.get("arm"), row.get("trial")
            if (not isinstance(task, str) or not task or not isinstance(arm, str) or not arm
                    or isinstance(trial, bool) or not isinstance(trial, int)):
                raise ValueError("cell identity is missing or malformed")
            if path.name != f"{task}__{arm}__t{trial}.json":
                raise ValueError("cell ledger filename does not match its identity")
            lifetime_billed_usd(row)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid cell cost record {path}: {exc}") from exc
        rows.append(row)
    return rows


def ledger_billed_usd(directory: Path) -> float:
    """Total recorded lifetime spend for every cell, including paid failures."""
    try:
        return math.fsum(lifetime_billed_usd(row) for row in read_cost_rows(directory))
    except OverflowError as exc:
        raise ValueError("ledger lifetime cost overflowed") from exc
