from list_utils import chunked, head, tail


def test_head():
    assert head([1, 2, 3]) == 1
    assert head([]) is None


def test_tail():
    assert tail([1, 2, 3]) == [2, 3]
    assert tail([]) == []


def test_chunked():
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunked([], 2) == []
