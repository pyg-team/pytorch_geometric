import numbers
import itertools


def repeat(src, length):
    if isinstance(src, numbers.Number):
        src = list(itertools.repeat(src, length))
    return src
