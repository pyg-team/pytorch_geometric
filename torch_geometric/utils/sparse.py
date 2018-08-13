from torch_sparse import coalesce


def dense_to_sparse(tensor):
    index = tensor.nonzero()
    value = tensor[index]
    index = index.t().contiguous()
    index, value = coalesce(index, value, tensor.size(0), tensor.size(1))
    return index, value
