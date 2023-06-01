from typing import Any, List, Union

import numpy
import torch


def tensor_equal_with_device(lhs: torch.Tensor, rhs: torch.Tensor):
    r""" Check whether two tensors are same for the data and device.
    """
    if lhs.device == rhs.device:
        return torch.equal(lhs, rhs)
    return False


def id2idx(ids: Union[List[int], torch.Tensor]):
    r""" id mapping between global ids and local index
    """
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.int64)
    max_id = torch.max(ids).item()
    id2idx = torch.zeros(max_id + 1, dtype=torch.int64, device=ids.device)
    id2idx[ids] = torch.arange(ids.size(0), dtype=torch.int64,
                               device=ids.device)
    return id2idx


def convert_to_tensor(data: Any, dtype: torch.dtype = None):
    r""" Convert input data intto a tensor based type.
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            new_data[k] = convert_to_tensor(v, dtype)
        return new_data
    if isinstance(data, list):
        new_data = []
        for v in data:
            new_data.append(convert_to_tensor(v, dtype))
        return new_data
    if isinstance(data, tuple):
        return tuple(convert_to_tensor(list(data), dtype))
    if isinstance(data, torch.Tensor):
        return data.type(dtype) if dtype is not None else data
    if isinstance(data, numpy.ndarray):
        return (torch.from_numpy(data).type(dtype)
                if dtype is not None else torch.from_numpy(data))
    return data


def apply_to_all_tensor(data: Any, tensor_method, *args, **kwargs):
    r""" Apply the method to all tensors contained inside input data recursively.
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            new_data[k] = apply_to_all_tensor(v, tensor_method, *args,
                                              **kwargs)
        return new_data
    if isinstance(data, list):
        new_data = []
        for v in data:
            new_data.append(
                apply_to_all_tensor(v, tensor_method, *args, **kwargs))
        return new_data
    if isinstance(data, tuple):
        return tuple(
            apply_to_all_tensor(list(data), tensor_method, *args, **kwargs))
    if isinstance(data, torch.Tensor):
        return tensor_method(data, *args, **kwargs)
    return data


def squeeze(data: Any):
    r""" Squeeze all tensors inside input data.
    """
    return apply_to_all_tensor(data, torch.Tensor.squeeze)
