"""The brain layer: LLM processes hosted on the Claude Agent SDK.

Everything here sits on :mod:`taste.memstore`, which is the only deterministic
part of the system. Nothing in this package may be imported by memstore.

The SDK is an optional dependency: importing this package without
``claude-agent-sdk`` installed raises a clear error rather than a confusing
``ModuleNotFoundError`` from three frames down.
"""

from __future__ import annotations

__all__ = ["MemstoreSessionStore"]


def __getattr__(name: str):
    if name == "MemstoreSessionStore":
        from taste.brains.session_store import MemstoreSessionStore

        return MemstoreSessionStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
