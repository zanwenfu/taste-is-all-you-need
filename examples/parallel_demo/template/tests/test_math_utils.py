from math_utils import add, clamp, mul


def test_add():
    assert add(2, 3) == 5


def test_mul():
    assert mul(3, 4) == 12


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10
