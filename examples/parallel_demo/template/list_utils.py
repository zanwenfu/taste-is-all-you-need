def head(xs):
    return xs[0] if xs else None


def tail(xs):
    return xs[1:] if xs else []


def chunked(xs, n):
    return [xs[i : i + n] for i in range(0, len(xs), n)]
