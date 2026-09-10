"""CI must actually run the brain layer, not skip it and report green.

``claude-agent-sdk`` is an optional extra, and every brain test begins with
``importorskip``. That is right for a contributor who has not installed it --
and silently wrong for CI, where it turned 42 tests (the sub-brain loop, the
kill proof, the whole monitor ladder) into skips while the run reported
success. Verified on a clean ``.[dev]`` venv before this guard existed.

A skip that hides the tests you most care about should be loud, so this asserts
the workflow installs the extra. It is a test about the build, deliberately, so
that the failure lands where the mistake is.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


@pytest.mark.skipif(not WORKFLOW.exists(), reason="no CI workflow in this checkout")
def test_ci_installs_the_optional_brain_dependency() -> None:
    text = WORKFLOW.read_text()
    installs = re.findall(r"pip install -e '\.\[([^\]]+)\]'", text)
    assert installs, f"no editable install found in {WORKFLOW}"
    for extras in installs:
        names = {e.strip() for e in extras.split(",")}
        assert "brains" in names, (
            f"CI installs .[{extras}] without 'brains', so claude-agent-sdk is "
            "absent and every brain test skips while the run reports green"
        )


@pytest.mark.skipif(not WORKFLOW.exists(), reason="no CI workflow in this checkout")
def test_ci_lints_what_we_lint_locally() -> None:
    """Two pushes failed because lint ran only on CI. If the scopes drift, the
    same thing happens again."""
    text = WORKFLOW.read_text()
    assert "ruff check taste/ tests/ examples/" in text, (
        "CI's ruff scope changed; keep it identical to what is run locally"
    )
