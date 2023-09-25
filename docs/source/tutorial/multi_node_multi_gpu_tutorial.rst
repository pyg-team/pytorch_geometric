Multi-Node-Multi-GPU GNN Training
==================================

Before doing this tutorial we recommend going through <insert single-node-multi-gpu tutorial> as a warm up.
Our first step is to understand the basic structure of a multi-node-multi-gpu example.


.. code-block:: python

    import argparse
    import os
    import time
    import warnings

    import torch
    import torch.distributed as dist
    from torch_geometric.datasets import FakeDataset

    from torch_geometric.nn.models import GCN

    warnings.filterwarnings("ignore")


    _LOCAL_PROCESS_GROUP = None


    def create_local_process_group(num_workers_per_node):
        ...


    def get_local_process_group():
        ...


    def run_train(device, data, world_size, model, epochs, batch_size, fan_out,
                  split_idx, num_classes):
        local_group = get_local_process_group()
        loc_id = dist.get_rank(group=local_group)
        rank = torch.distributed.get_rank()
        os.environ['NVSHMEM_SYMMETRIC_SIZE'] = "107374182400"
        # run training
        ...


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_channels', type=int, default=64)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--fan_out', type=int, default=50)
        parser.add_argument(
            "--ngpu_per_node",
            type=int,
            default="1",
            help="number of GPU(s) for each node for multi-gpu training,",
        )
        args = parser.parse_args()
        # setup multi node
        torch.distributed.init_process_group("nccl")
        nprocs = dist.get_world_size()
        create_local_process_group(args.ngpu_per_node)
        local_group = get_local_process_group()
        device_id = dist.get_rank(
            group=local_group) if dist.is_initialized() else 0
        torch.cuda.set_device(device_id)
        device = torch.device(device_id)

        dataset = FakeDataset(avg_num_nodes=100000)
        data = dataset.data
        num_nodes = data.num_nodes
        rand_id = torch.randperm(num_nodes)

        # 60/20/20 split
        split_idx = {
            'train':rand_id[:int(.6 * num_nodes)],
            'valid':rand_id[int(.6 * num_nodes):int(.8 * num_nodes)],
            'test':rand_id[:int(.8 * num_nodes):],
        }

        model = GCN(dataset.num_features, args.hidden_channels, 2,
                    dataset.num_classes)
        run_train(device, data, nprocs, model, args.epochs, args.batch_size,
                  args.fan_out, split_idx, dataset.num_classes)

# TODO: explain code above then fill in ...s in next blocks



And that's it.
Putting it all together gives a working multi-node-multi-GPU example that follows a training flow that is similar to single GPU training.
You can run the shown tutorial by yourself by looking at `examples/multi_gpu/multi_node_multi_gpu_synthetic.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/multi_node_multi_gpu_synthetic.py>`_.