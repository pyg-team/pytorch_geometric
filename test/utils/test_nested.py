import torch

from torch_geometric.utils import from_nested_tensor, to_nested_tensor


def test_to_and_from_nested_tensor():
    x = torch.randn(5, 5)
    batch = torch.tensor([0, 0, 1, 1, 1])

    nested = to_nested_tensor(x, batch)
    # print(nested)
    # print(nested.is_nested)
    # print(out)
    # # print(out.__class__.__dict__)
    # print(out.__class__)
    # print(out.storage()._storage.__dict__)
    # out = out.contiguous()
    # # print(out.storage().__dict__)

    # bla = torch.tensor(out.storage())
    # print(bla)
    # print(bla.shape)

    from_nested_tensor(nested)
