import torch.sparse as sparse
import torch.cuda.sparse as cuda_sparse


def SparseTensor(indices, values, size):
    if values.is_cuda:
        A = getattr(cuda_sparse, values.__class__.__name__)
    else:
        A = getattr(sparse, values.__class__.__name__)

    return A(indices, values, size)
