"""A small policy tree that picks a recovery from what a harness can observe.

Each failed run has features a deployed harness sees without rerunning it
(its length, the checker's confidence, whether the visible tests ever passed)
and, for every recovery, the outcome measured from the same failed state: the
share of its repeats the hidden tests passed. A tree splits the runs on one
feature at a time and gives each leaf the recovery with the best mean outcome
among its runs (ties to the cheaper, then by name). A split is kept only when
it raises the leaves' total outcome and leaves ``min_leaf`` runs on each side.

``cross_validate`` scores it on runs it was not fitted on: each run's recovery
is chosen by a tree fitted without the run's fold. With depth 0 the tree is
the best fixed recovery on the training runs, the fair baseline for a
selector.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

TOLERANCE = 1e-12


@dataclass(frozen=True)
class Leaf:
    arm: str
    runs: int


@dataclass(frozen=True)
class Split:
    feature: str
    threshold: float
    left: object                # runs with the feature at most the threshold
    right: object


def value(features, name):
    raw = features[name]
    return float(raw) if not isinstance(raw, bool) else float(int(raw))


def _mean(rows, key, arm):
    return math.fsum(row[key][arm] for row in rows) / len(rows) if rows else 0.0


def best_arm(rows, arms):
    """The arm with the best mean outcome over the rows; ties to the cheaper, then by name."""
    return min(arms, key=lambda arm: (-_mean(rows, "outcomes", arm),
                                      _mean(rows, "costs", arm) if all("costs" in row for row in rows) else 0.0,
                                      arm))


def _total(rows, arm):
    return math.fsum(row["outcomes"][arm] for row in rows)


def fit(rows, arms, features, *, depth=2, min_leaf=5):
    """A policy tree over the rows: {"features": {...}, "outcomes": {arm: x}, "costs": {arm: $}}."""
    chosen = best_arm(rows, arms)
    leaf = Leaf(chosen, len(rows))
    if depth <= 0 or len(rows) < 2 * min_leaf:
        return leaf
    best = None
    for feature in features:
        levels = sorted({value(row["features"], feature) for row in rows})
        for low, high in itertools.pairwise(levels):
            threshold = (low + high) / 2
            left = [row for row in rows if value(row["features"], feature) <= threshold]
            right = [row for row in rows if value(row["features"], feature) > threshold]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            gain = _total(left, best_arm(left, arms)) + _total(right, best_arm(right, arms))
            if best is None or gain > best[0] + TOLERANCE:
                best = (gain, feature, threshold, left, right)
    if best is None or best[0] <= _total(rows, chosen) + TOLERANCE:
        return leaf
    _, feature, threshold, left, right = best
    return Split(feature, threshold, fit(left, arms, features, depth=depth - 1, min_leaf=min_leaf),
                 fit(right, arms, features, depth=depth - 1, min_leaf=min_leaf))


def choose(tree, features):
    while isinstance(tree, Split):
        tree = tree.left if value(features, tree.feature) <= tree.threshold else tree.right
    return tree.arm


def describe(tree, indent=""):
    """The tree as indented lines of text."""
    if isinstance(tree, Leaf):
        return [f"{indent}-> {tree.arm} ({tree.runs} runs)"]
    return [f"{indent}if {tree.feature} <= {tree.threshold:g}:", *describe(tree.left, indent + "  "),
            f"{indent}else:", *describe(tree.right, indent + "  ")]


def cross_validate(rows, arms, features, *, folds=5, depth=2, min_leaf=5, seed=0):
    """The arm chosen for each row by a tree fitted on the other folds (seeded fold assignment)."""
    if len(rows) < 2:
        return [best_arm(rows, arms) for _ in rows]
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    folds = max(2, min(folds, len(rows)))
    fold_of = {index: position % folds for position, index in enumerate(order)}
    chosen = [None] * len(rows)
    for fold in range(folds):
        training = [row for index, row in enumerate(rows) if fold_of[index] != fold]
        tree = fit(training, arms, features, depth=depth, min_leaf=min_leaf)
        for index, row in enumerate(rows):
            if fold_of[index] == fold:
                chosen[index] = choose(tree, row["features"])
    return chosen
