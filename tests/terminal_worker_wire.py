"""Network-only worker fixture, safe to import with Claude SDK imports denied."""

import json

from tests.test_azure_worker_runtime import accepted
from tests.test_openai_responses import function_call, message


def replies(number, payload):
    if number == 1:
        return [function_call(json.dumps({"command": "produce evidence", "cwd": "/tmp", "timeout_seconds": 3}),
                              name="terminal_exec", call_id="terminal_call")]
    if number == 2:
        return [function_call(json.dumps({"path": "output.txt", "content": "correct", "executable": False}),
                              name="write_artifact", call_id="artifact_call")]
    return [message(json.dumps(accepted(payload)))]
