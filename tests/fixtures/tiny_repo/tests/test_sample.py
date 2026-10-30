"""Independent behavioral checks include both failing and preserved behavior."""

from sample import double, increment, is_even


def test_increment_zero():
    assert increment(0) == 1


def test_increment_negative():
    assert increment(-2) == -1


def test_double_positive():
    assert double(3) == 6


def test_double_zero():
    assert double(0) == 0


def test_even():
    assert is_even(4)


def test_odd():
    assert not is_even(3)
