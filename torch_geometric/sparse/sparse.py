import torch.sparse as sparse
import torch.cuda.sparse as cuda_sparse


def SparseTensor(indices, values, size):
    if values.is_cuda:
        A = getattr(cuda_sparse, values.type().split('.')[-1])
    else:
        print(values)
        A = getattr(sparse, values.type().split('.')[-1])

    return A(indices, values, size)
