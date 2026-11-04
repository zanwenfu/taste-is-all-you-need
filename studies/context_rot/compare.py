"""Comparisons for the rule-loss study: rates with intervals, paired arms, repeated rounds, judges.

Rates leave out unreadable answers (None), as the package's summary does, and say how many.
Intervals: Wilson for a rate; a history-cluster bootstrap where the 15 rules of one history
are not independent cases.

    python -m studies.context_rot.compare arms <baseline folder> <arm folder> [<arm folder> ...]
    python -m studies.context_rot.compare repeated <baseline folder> <repeated folder> [...]
    python -m studies.context_rot.compare judges <judged results> <other verdicts>
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (centre - half), 1), round(100 * (centre + half), 1)


def rate(rows, key: str) -> dict:
    """A rate for display ("pct", one decimal) and its exact value ("exact") for any further arithmetic."""
    values = [row[key] for row in rows if row.get(key) is not None]
    hits = sum(bool(v) for v in values)
    exact = 100 * hits / len(values) if values else None
    return {"pct": None if exact is None else round(exact, 1), "exact": exact, "n": len(values),
            "unreadable": len(rows) - len(values), "wilson": wilson(hits, len(values))}


def median(values) -> float | None:
    """The usual median: the mean of the two middle values when the count is even."""
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def effective_retention(compacted: float, floor: float, ceiling: float) -> float | None:
    """The paper's measure: how much of the gap between no rule and the rule restated is recovered."""
    if ceiling == floor:
        return None
    return 100 * (compacted - floor) / (ceiling - floor)


def paired(base: dict, other: dict, base_key: str = "compaction_compliant",
           other_key: str = "compaction_compliant") -> dict:
    """Same cases, two contexts: changes in compliance and an exact two-sided sign test."""
    better = worse = same = unreadable = 0
    for case, row in base.items():
        a, b = row.get(base_key), other[case].get(other_key)
        if a is None or b is None:
            unreadable += 1
        elif a == b:
            same += 1
        elif b:
            better += 1
        else:
            worse += 1
    n = better + worse
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n) if n else 1.0
    return {"better": better, "worse": worse, "same": same, "unreadable": unreadable, "sign_test_p": p}


def cluster_interval(rows, key: str, cluster: str = "haystack_id", draws: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    """95% interval of a rate, resampling whole histories with replacement."""
    groups: dict[str, list[bool]] = {}
    for row in rows:
        if row.get(key) is not None:
            groups.setdefault(row[cluster], []).append(bool(row[key]))
    names = sorted(groups)
    rng = random.Random(seed)
    rates = []
    for _ in range(draws):
        picked = [groups[rng.choice(names)] for _ in names]
        total = sum(len(g) for g in picked)
        rates.append(100 * sum(sum(g) for g in picked) / total)
    rates.sort()
    return round(rates[int(0.025 * draws)], 1), round(rates[min(draws - 1, int(0.975 * draws))], 1)


ARMS = ("compaction", "user_record", "recent_control", "criteria")


def repeated_table(rows, baseline: dict) -> dict:
    """Per prompt and k: rule kept by round, compliance per context, and effective retention.

    The floor (no rule in the history at all) is the baseline's full-history probe without the
    rule, on the same cases; it does not depend on k.
    """
    table: dict = {}
    for row in rows:
        table.setdefault(row["prompt"], {}).setdefault(row["k"], []).append(row)
    out: dict = {}
    for prompt, by_k in table.items():
        for k, group in sorted(by_k.items()):
            rounds = len(group[0]["retention"])
            kept_by_round = []
            for r in range(rounds):
                verdicts = [g["retention"][r] for g in group if g["retention"][r] is not None]
                kept_by_round.append(round(100 * sum(verdicts) / len(verdicts), 1) if verdicts else None)
            floor_rate = rate([baseline[g["id"]] for g in group], "full_without_sc_compliant")
            floor = floor_rate["exact"]
            entry = {"cases": len(group), "kept_by_round_pct": kept_by_round,
                     "kept_after_last_round_pct": kept_by_round[-1], "floor_pct": floor_rate["pct"]}
            for arm in [a for a in ARMS if f"{a}_compliant" in group[0]]:
                compliant = rate(group, f"{arm}_compliant")
                upper = rate(group, f"{arm}_upper_compliant")
                entry[f"{arm}_pct"] = compliant["pct"]
                entry[f"{arm}_wilson"] = compliant["wilson"]
                entry[f"{arm}_unreadable"] = compliant["unreadable"]
                entry[f"{arm}_upper_pct"] = upper["pct"]
                er = effective_retention(compliant["exact"], floor, upper["exact"]) if None not in (
                    compliant["exact"], floor, upper["exact"]) else None
                entry[f"{arm}_effective_retention"] = None if er is None else round(er, 1)
            out.setdefault(prompt, {})[k] = entry
    return out


def judge_agreement(a: dict, b: dict) -> dict:
    """Two judges' verdicts on the same cases: agreement and Cohen's kappa where both are clear."""
    both = [(a[case], b[case]) for case in a if case in b and a[case] is not None and b[case] is not None]
    n = len(both)
    if not n:
        return {"both_clear": 0, "agree_pct": None, "kappa": None}
    observed = sum(x == y for x, y in both) / n
    pa, pb = sum(x for x, _ in both) / n, sum(y for _, y in both) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {"both_clear": n, "agree_pct": 100 * observed, "kappa": kappa,
            "yes_pct": (100 * pa, 100 * pb), "a_only_yes": sum(x and not y for x, y in both),
            "b_only_yes": sum(y and not x for x, y in both)}


def _load(folder: str | Path, name: str = "results.jsonl") -> dict:
    path = Path(folder) / name
    if not path.exists() and name == "results.judged.jsonl":
        path = Path(folder) / "results.jsonl"
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    command, rest = argv[0], argv[1:]
    if command == "arms":
        base = _load(rest[0], "results.judged.jsonl")
        report = {"baseline": {key: rate(list(base.values()), f"{key}_compliant")
                               for key in ("full_with_sc", "full_without_sc", "compaction", "upper_bound")}}
        report["baseline"]["compaction_cluster_interval"] = cluster_interval(list(base.values()), "compaction_compliant")
        for folder in rest[1:]:
            arm = _load(folder)
            report[Path(folder).name] = {"compaction": rate(list(arm.values()), "compaction_compliant"),
                                         "upper_bound": rate(list(arm.values()), "upper_bound_compliant"),
                                         "cluster_interval": cluster_interval(list(arm.values()),
                                                                              "compaction_compliant"),
                                         "vs_baseline": paired(base, arm)}
    elif command == "repeated":
        base = _load(rest[0], "results.judged.jsonl")
        rows = [row for folder in rest[1:] for row in map(json.loads, (Path(folder) / "results.jsonl")
                                                          .read_text().splitlines())]
        report = repeated_table(rows, base)
    elif command == "judges":
        a = {case: row.get("retention") for case, row in _load(rest[0], "results.judged.jsonl").items()}
        b = {item["id"]: item["retention"] for item in map(json.loads, Path(rest[1]).read_text().splitlines())}
        report = judge_agreement(a, b)
    else:
        raise SystemExit(f"unknown command {command}")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
