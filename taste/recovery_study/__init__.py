"""The recovery study's drivers: where failed runs become unwinnable, and which recovery works.

The study (an internal design, docs/design_recovery_study.md) branches an
unmodified agent's failed run from its recorded steps to find the point of no
return, then compares four recoveries from the same failed state at equal
dollars and wall-clock. This package drives it and analyses it:

- ``jobs``: every Harbor command the study starts (``infra/azure/run-harbor.sh``
  with ``--ak`` settings), and the names of the settings branching (work
  package A) provides, in one table.
- ``records``: what a finished trial says, from Harbor's job directories
  (reward, settings, what it handed back) and Taste's settled records (cost,
  tokens, time, steps).
- ``search``: the binary search for the point of no return, as pure functions
  of the probes' results, and the curve it leaves.
- ``rules`` and ``selector``: rewind points chosen without reruns, and a small
  policy tree that picks a recovery from what a harness can observe.
- ``map_driver`` and ``recovery_driver``: resumable drivers. Each run reads the
  jobs it started, decides the next ones and writes their command lines; its
  state is one JSON file.

The checker and the trajectory reader are ``taste.agents.checker`` and
``taste.agents.trajectory_reader``: their task text, verdicts, feedback, step
numbering and rules are used as they are. Nothing imported needs a package
beyond the standard library, so the drivers run with any Python 3.11 or later
that can import ``taste`` and read the records.
"""
