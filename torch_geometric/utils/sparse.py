from .coalesce import coalesce


def dense_to_sparse(tensor):
    index = tensor.nonzero()
    value = tensor[index]
    index = index.t().contiguous()
    index, value = coalesce(index, value)
    return index, value
