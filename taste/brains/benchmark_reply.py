"""Opt-in final developer replies, bound to the immutable benchmark goal.

Status/completion reasons and worker summaries are internal evidence. Only an
accepted coordinator proposal may supply the final reply shown to a benchmark.
The existing Goal/Plan wire formats remain compatible for non-benchmark goals.
"""

KEY = "benchmark_reply"
SCHEMA = "taste.brains/BenchmarkReply/1"
MAX_REPLY_BYTES = 128 * 1024


def required(metadata):
    if KEY not in metadata:
        return False
    if metadata[KEY] != SCHEMA:
        raise ValueError("unsupported benchmark reply contract")
    return True


def validate(metadata, *, complete):
    reply = metadata.get("final_reply")
    if (not isinstance(reply, str) or "\x00" in reply
            or len(reply.encode("utf-8")) > MAX_REPLY_BYTES):
        raise ValueError("benchmark final_reply must be text within 128 KiB")
    if (complete and not reply.strip()) or (not complete and reply != ""):
        raise ValueError("only a complete benchmark proposal must contain a final_reply")
    return reply
