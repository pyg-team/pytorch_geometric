import warnings
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

        You need to use the :class:`torch_geometric.data.DataListLoader` for
        this module.

    Args:
        module (Module): Module to be parallelized.
        device_ids (list of int or torch.device): CUDA devices.
            (default: all devices)
        output_device (int or torch.device): Device location of output.
            (default: :obj:`device_ids[0]`)
    """

    def __init__(self, module, device_ids=None, output_device=None,
                 use_original=True):
        super(DataParallel, self).__init__(module, device_ids, output_device)
        self.src_device = torch.device("cuda:{}".format(self.device_ids[0]))

        # To store the mapping back to the input order
        self.id_map = []

        # Use the original algorithm
        self.use_original = use_original

    def forward(self, data_list):
        """"""
        if len(data_list) == 0:
            warnings.warn('DataParallel received an empty data list, which '
                          'may result in unexpected behaviour.')
            return None

        if not self.device_ids or len(self.device_ids) == 1:  # Fallback
            data = Batch.from_data_list(data_list).to(self.src_device)
            return self.module(data)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    ('Module must have its parameters and buffers on device '
                     '{} but found one of them on device {}.').format(
                        self.src_device, t.device))

        inputs = self.scatter(data_list, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

    def scatter(self, data_list, device_ids):
        r"""
        This function distributes the data in the data_list into batches
        to be sent to the gpus in device_ids
         - This first spreads the largest graphs to the gpus in order
         - Then it add the next largest graph to the gpu with the lowest number
          of nodes
         - Finally it add the smallest graphs to gpus to ensure each gpu has at
          least 2 graphs
        """
        # Gets the number of GPUs being used
        num_devices = min(len(device_ids), len(data_list))

        # Gets the number of nodes in each data object in the input data list
        num_nodes = torch.tensor([data.num_nodes for data in data_list])

        if self.use_original:
            # The original implementation
            cumsum = num_nodes.cumsum(0)
            cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
            device_id = num_devices * cumsum.to(torch.float) / cumsum[
                -1].item()
            device_id = (device_id[:-1] + device_id[1:]) / 2.0
            device_id = device_id.to(torch.long)  # round.
            split = device_id.bincount().cumsum(0)
            split = torch.cat([split.new_zeros(1), split], dim=0)
            split = torch.unique(split, sorted=True)
            split = split.tolist()

            return [
                Batch.from_data_list(data_list[split[i]:split[i + 1]]).to(
                    torch.device('cuda:{}'.format(device_ids[i])))
                for i in range(len(split) - 1)
            ]

        else:
            # The proposed implementation
            sorted_indices = torch.argsort(num_nodes, dim=0, descending=True)

            # This will contain the GPU the data object has been assigned.
            # It is initialised with the num_device largest node counts
            # This in reverse order so GPU: 0 hopefully ends up with less nodes
            new_device_id = torch.zeros(len(data_list), dtype=torch.long)

            # These tensors track how many nodes and how many graphs are on 
            # each GPU respectively
            num_nodes_per_device = torch.tensor([0] * num_devices)
            num_graphs_per_device = torch.tensor([0] * num_devices)

            # This is the main loop that does the assignment of the data 
            # objects to the devices
            for ii in range(num_nodes.size()[0] - num_devices):
                if ii < num_devices:
                    # Fill the tracker with the new_device_id initialised 
                    # values
                    gpu_id = num_devices - ii - 1
                    new_device_id[ii] = gpu_id
                    num_nodes_per_device[new_device_id[ii]] += num_nodes[
                        sorted_indices[ii]]
                    num_graphs_per_device[new_device_id[ii]] += 1
                else:
                    # This assigns the data object to the device with the 
                    # least nodes
                    new_device_id[ii] = torch.argmin(
                        num_nodes_per_device).item()
                    num_graphs_per_device[new_device_id[ii]] += 1
                    num_nodes_per_device[new_device_id[ii]] += num_nodes[
                        sorted_indices[ii]]

            # This checks the last num_device data objects in the list 
            # (smallest ones)
            # and part ensures that there are at least 2 graphs per device
            # This is in reverse order to put the smallest remaining graph 
            # on the GPU with the most nodes
            for ii in range(num_nodes.size()[0] - 1,
                            num_nodes.size()[0] - num_devices - 1, -1):
                if num_graphs_per_device[
                    torch.argmin(num_graphs_per_device).item()] == 1:
                    mask = num_graphs_per_device == 1
                    gpu_id = \
                        torch.masked_select(torch.arange(num_devices), mask)[
                            torch.argmax(
                                torch.masked_select(num_nodes_per_device,
                                                    mask))]
                    new_device_id[ii] = gpu_id.item()
                    num_graphs_per_device[new_device_id[ii]] += 1
                    num_nodes_per_device[new_device_id[ii]] += num_nodes[
                        sorted_indices[ii]]
                else:
                    # If they all have 2 or more graphs it assigns the it to 
                    # the device with the least nodes
                    new_device_id[ii] = torch.argmin(
                        num_nodes_per_device).item()
                    num_graphs_per_device[new_device_id[ii]] += 1
                    num_nodes_per_device[new_device_id[ii]] += num_nodes[
                        sorted_indices[ii]]

            # Now the assigned data objects are converted in to batches for 
            # each device
            # The id_map is filled in at this point. It must be emptied at 
            # each call of scatter
            list_of_batches = []
            self.id_map = []
            for gpu in range(num_devices):
                tmp_data_list = []
                for jj, gpu_id in enumerate(new_device_id):
                    if gpu_id == gpu:
                        tmp_data_list.append(data_list[sorted_indices[jj]])
                        self.id_map.append(sorted_indices[jj].item())

                list_of_batches.append(Batch.from_data_list(tmp_data_list).to(
                    torch.device('cuda:{}'.format(gpu))))

            return list_of_batches

    def gather(self, outputs, output_device):
        """
        Gathers the results from the GPUs and returns the answer in the correct 
        order
        """
        if self.use_original:
            return super().gather(outputs, output_device)
        else:
            return torch.index_select(super().gather(outputs, output_device),
                                      0,
                                      torch.argsort(
                                          torch.LongTensor(self.id_map).to(
                                              output_device), dim=0))
