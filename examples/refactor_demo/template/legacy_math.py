def run(items):
    total = 0
    for x in items:
        if x > 0:
            total += x * 2
        else:
            total -= x
    return total


def fmt(total):
    return f"total is {total}"


def main(items):
    return fmt(run(items))
