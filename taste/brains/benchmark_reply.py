"""Opt-in final developer replies, bound to the immutable benchmark goal.

Status/completion reasons and worker summaries are internal evidence. Only an
accepted coordinator proposal may supply the final reply shown to a benchmark.
The existing Goal/Plan wire formats remain compatible for non-benchmark goals.
"""

KEY = "benchmark_reply"
SCHEMA = "taste.brains/BenchmarkReply/1"
MAX_REPLY_BYTES = 128 * 1024
# Seconds held back from the goal's working time. A run that stops short of
# completion spends them asking the coordinator once for its final reply, as
# a benchmark's own reference agent is asked for its report when time is up.
RESERVE_KEY = "benchmark_reply_reserve_seconds"
# The least working time in which another plan may still be started. A plan
# asked for in the last seconds cannot be answered, let alone acted on.
PLAN_KEY = "benchmark_reply_plan_seconds"
CLOSING_OPERATION_PREFIX = "runtime-closing."


def required(metadata):
    if KEY not in metadata:
        if RESERVE_KEY in metadata:
            raise ValueError("a reply reserve requires the benchmark reply contract")
        return False
    if metadata[KEY] != SCHEMA:
        raise ValueError("unsupported benchmark reply contract")
    closing_reserve(metadata)
    planning_minimum(metadata)
    return True


def closing_reserve(metadata):
    """Seconds reserved for a closing reply; 0 when a goal asks for none."""
    if KEY not in metadata or RESERVE_KEY not in metadata:
        return 0.0
    value = metadata[RESERVE_KEY]
    if type(value) not in (int, float) or not 10 <= value <= 3600:
        raise ValueError("benchmark reply reserve must be 10-3600 seconds")
    return float(value)


def planning_minimum(metadata):
    """Seconds of working time a new plan needs; 0 when a goal names none."""
    if KEY not in metadata or PLAN_KEY not in metadata:
        return 0.0
    value = metadata[PLAN_KEY]
    if type(value) not in (int, float) or not 0 <= value <= 3600:
        raise ValueError("benchmark planning minimum must be 0-3600 seconds")
    return float(value)


def is_closing(operation_id):
    return isinstance(operation_id, str) and operation_id.startswith(CLOSING_OPERATION_PREFIX)


def validate(metadata, *, complete, closing=False):
    reply = metadata.get("final_reply")
    if (not isinstance(reply, str) or "\x00" in reply
            or len(reply.encode("utf-8")) > MAX_REPLY_BYTES):
        raise ValueError("benchmark final_reply must be text within 128 KiB")
    if closing and not reply.strip():
        raise ValueError("a closing benchmark proposal must contain its final_reply")
    if not closing and ((complete and not reply.strip()) or (not complete and reply != "")):
        raise ValueError("only a complete benchmark proposal must contain a final_reply")
    return reply
