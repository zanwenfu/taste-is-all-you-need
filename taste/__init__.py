"""Agent OS: a git-native harness for long-running agents.

Public surface:
    from taste import Kernel, Memory, agent, tool
"""

from taste.agent import AgentSpec, agent
from taste.kernel import Kernel, RunResult
from taste.memory import Checkpoint, Memory
from taste.tools import Tool, tool

__all__ = [
    "AgentSpec",
    "Checkpoint",
    "Kernel",
    "Memory",
    "RunResult",
    "Tool",
    "agent",
    "tool",
]

__version__ = "0.1.0"
