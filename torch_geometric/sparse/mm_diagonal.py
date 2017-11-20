from .sparse import SparseTensor


def mm_diagonal(a, b, transpose=False):
    index = b._indices()
    target = index[1] if transpose else index[0]
    value = b._values()

    value = a[target] * value

    return SparseTensor(index, value, b.size())
