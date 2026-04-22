from legacy_math import fmt, main, run


def test_run_positive_only():
    assert run([1, 2, 3]) == 12


def test_run_mixed_signs():
    assert run([1, -2, 3]) == 10


def test_run_empty():
    assert run([]) == 0


def test_fmt():
    assert fmt(7) == "total is 7"


def test_main_roundtrip():
    assert main([1, 2]) == "total is 6"
