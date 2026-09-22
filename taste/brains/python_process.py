"""Import our process entrypoints from installed code, independent of task files."""
from collections.abc import Sequence
from pathlib import Path


def isolated_python_argv(executable: str, code: str, arguments: Sequence[str]) -> list[str]:
    # -I excludes the task cwd, ambient PYTHONPATH and user site packages.
    # Explicitly bind the code root: editable installs in a testing venv may
    # refer to another checkout. Only this known root is inserted into sys.path.
    root = str(Path(__file__).resolve().parents[2])
    return [executable, "-I", "-c",
            "import sys; sys.path.insert(0, sys.argv.pop(1)); " + code, root, *arguments]
