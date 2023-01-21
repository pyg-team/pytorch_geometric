import logging
from itertools import chain

import torch

from torch_geometric.data import Batch


class DataParallel(torch.nn.DataParallel):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting a list of :class:`torch_geometric.data.Data` objects and copying
    them as :class:`torch_geometric.data.Batch` objects to each device.
    In the forward pass, the module is replicated on each device, and each
    replica handles a portion of the input.
    During the backwards pass, gradients from each replica are summed into the
    original module.

    The batch size should be larger than the number of GPUs used.

    The parallelized :attr:`module` must have its parameters and buffers on
    :obj:`device_ids[0]`.

    .. note::

        You need to use the :class:`torch_geometric.loader.DataListLoader` for
        this module.

    Args:
        module (Module): Module to be parallelized.
        device_ids (list of int or torch.device): CUDA devices.
            (default: all devices)
        output_device (int or torch.device): Device location of output.
            (default: :obj:`device_ids[0]`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (list or tuple, optional): Will exclude each key in the
            list. (default: :obj:`None`)
    """
    def __init__(self, module, device_ids=None, output_device=None,
                 follow_batch=None, exclude_keys=None):
        super().__init__(module, device_ids, output_device)
        self.src_device = torch.device(f'cuda:{self.device_ids[0]}')
        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []

    def forward(self, data_list):
        """"""
        if len(data_list) == 0:
            logging.warning('DataParallel received an empty data list, which '
                            'may result in unexpected behaviour.')
            return None

        if not self.device_ids or len(self.device_ids) == 1:  # Fallback
            data = Batch.from_data_list(
                data_list, follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys).to(self.src_device)
            return self.module(data)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    f"Module must have its parameters and buffers on device "
                    f"'{self.src_device}' but found one of them on device "
                    f"'{t.device}'")

        inputs = self.scatter(data_list, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

    def scatter(self, data_list, device_ids):
        num_devices = min(len(device_ids), len(data_list))

        count = torch.tensor([data.num_nodes for data in data_list])
        cumsum = count.cumsum(0)
        cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
        device_id = num_devices * cumsum.to(torch.float) / cumsum[-1].item()
        device_id = (device_id[:-1] + device_id[1:]) / 2.0
        device_id = device_id.to(torch.long)  # round.
        split = device_id.bincount().cumsum(0)
        split = torch.cat([split.new_zeros(1), split], dim=0)
        split = torch.unique(split, sorted=True)
        split = split.tolist()

        return [
            Batch.from_data_list(data_list[split[i]:split[i + 1]],
                                 follow_batch=self.follow_batch,
                                 exclude_keys=self.exclude_keys).to(
                                     torch.device(f'cuda:{device_ids[i]}'))
            for i in range(len(split) - 1)
        ]
