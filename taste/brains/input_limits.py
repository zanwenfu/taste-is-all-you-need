"""Bounded file/observation transport for full benchmark instructions.

Harbor instructions can approach 120 KiB before JSON escaping. Keep room for
that text and its immutable assignment/policy envelope. Model request budgets
remain separately admitted; these limits do not increase a provider allowance.
Tool outputs and cleanup receipts retain their smaller independent limits.
"""

MAX_GOAL_INPUT_BYTES = 512 * 1024
MAX_WORKER_OBSERVATION_BYTES = 512 * 1024
