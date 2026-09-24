"""Start blocking work whose owner must wait through event-loop shutdown.

asyncio.run cancels every Task before draining the default executor. A Task
wrapping to_thread can therefore finish while its thread still mutates state.
Keep an executor Future instead, and await it with asyncio.wait so caller
cancellation cannot cancel it. The caller still owns settlement and cleanup.
"""

from __future__ import annotations

import asyncio
import contextvars
from functools import partial
from typing import Any


def start_owned_thread(call, /, *args, **kwargs) -> asyncio.Future[Any]:
    context = contextvars.copy_context()

    def invoke():
        try:
            return context.run(partial(call, *args, **kwargs))
        except (KeyboardInterrupt, SystemExit) as exc:
            # These exceptions raised directly by a Task can stop the loop
            # before its owner gets to retrieve the failure and drain work.
            raise BaseExceptionGroup("owned thread interrupted", [exc]) from None

    return asyncio.get_running_loop().run_in_executor(None, invoke)
