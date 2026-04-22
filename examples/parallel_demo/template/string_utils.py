def upper(s):
    return s.upper()


def snake_case(s):
    out = []
    for i, ch in enumerate(s):
        if ch.isupper() and i:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def truncate(s, n):
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"
