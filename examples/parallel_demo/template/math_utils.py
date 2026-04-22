def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
