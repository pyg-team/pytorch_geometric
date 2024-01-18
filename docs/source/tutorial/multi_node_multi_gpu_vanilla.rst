Multi-Node Training using SLURM
===============================

This tutorial introduces a skeleton on how to perform distributed training on multiple GPUs over multiple nodes using the `SLURM workload manager <https://slurm.schedmd.com/>`_ available at many supercomputing centers.
The code is based on `our tutorial on single-node multi-GPU training <multi_gpu_vanilla.html>`_.
Please go there first to understand the basics if you are unfamiliar with the concepts of distributed training in :pytorch:`PyTorch`.

.. note::
    The complete script of this tutorial can be found at `examples/multi_gpu/distributed_sampling_multinode.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling_multinode.py>`_.
    You can find the example :obj:`*.sbatch` file `next to it <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling_multinode.sbatch>`_ and tune it to your needs.

A submission script to manage startup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we are now running on multiple nodes, we can no longer use our :obj:`__main__` entrypoint and start processes from there.
This is where the workload manager comes in as it allows us to prepare a special :obj:`*.sbatch` file.
This file is a standard bash script with instructions on how to setup the processes and your environment.

Our example starts with the usual shebang :obj:`#!/bin/bash` and special comments instructing which resources the SLURM system should reserve for our training run.
Configuration of the specifics usually depends on your site (and your usage limits!).
The following is a minimal example which works with a quite unrestricted configuration available to us:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=pyg-multinode-tutorial # identifier for the job listings
    #SBATCH --output=pyg-multinode.log        # outputfile
    #SBATCH --partition=gpucloud              # ADJUST this to your system
    #SBATCH -N 2                              # number of nodes you want to use
    #SBATCH --ntasks=4                        # number of processes to be run
    #SBATCH --gpus-per-task=1                 # every process wants one GPU!
    #SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding...

This example will create two processes each on two nodes with each process having a single GPU reserved.

In the following part, we have to set up some environment variables for :obj:`torch.distributed` to properly do the rendezvous procedure.
In theory you could also set those inside the :python:`Python` process:

.. code-block:: bash

    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

If you do not want to let your script randomly open a port and listen for incoming connections, you can also use a file on your shared filesystem.

Now the only thing left to add is the execution of the training script:

.. code-block:: console

    srun python distributed_sampling_multinode.py

Note how the :obj`python` call is prefixed with the :obj:`srun` command and thus :obj:`--ntasks` replicas will be started.

Finally, to submit the :obj:`*.sbatch` file itself into the work queue, use the :obj:`sbatch` utility in your shell:

.. code-block:: console

    sbatch distributed_sampling_multinode.sbatch

Using a cluster configured with pyxis-containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your cluster supports the :obj:`pyxis` plugin developed by NVIDIA, you can use a ready-to-use :pyg:`PyG` container that is updated each month with the latest from NVIDIA and :pyg:`PyG`, see `here <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg>`_ for more information.
The container sets up all necessary environment variables from which you can now directly run the example using :obj:`srun` from your command prompt:

.. code-block:: console

    srun --partition=<partitionname> -N<num_nodes> --ntasks=<number of GPUS in total> --gpus-per-task=1 --gpu-bind=none --container-name=pyg-test --container-image=<image_url> --container-mounts='.:/workspace' python3 distributed_sampling_multinode.py

Note that :obj:`--container-mounts='.:/workspace'` makes the current folder (which should include the example code) available in the default startup folder :obj:`workspace` of the container.

If you want to eventually customize packages in the container without having access to :obj:`docker` (very likely on a public HPC), you can create your own image by following `this tutorial <https://doku.lrz.de/9-creating-and-reusing-a-custom-enroot-container-image-10746637.html>`_.

Modifying the training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As SLURM now takes care of creating multiple :python:`Python` processes and we can not share any data (each process will have the full dataset loaded!), our :obj:`__main__` section now has to query the environment for the process setup generated by SLURM or the :obj:`pyxis` container:

.. code-block:: python

    # Get the world size from the WORLD_SIZE variable or directly from SLURM:
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    # Likewise for RANK and LOCAL_RANK:
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    run(world_size, rank, local_rank)

The :meth:`torch.distributed.init_process_group` function will now pick up the :obj:`MASTER_ADDR` from the environment:

.. code-block:: python

    def run(world_size: int, rank: int, local_rank: int):
        dist.init_process_group('nccl', world_size=world_size, rank=rank)

We also have to replace the usage of :obj:`rank` depending on whether we want to use it for node-local purposes like selecting a GPU or global tasks such as data splitting

.. code-block:: python

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

while we need to assign the model to a node-local GPU and thus use :obj:`local_rank`:

.. code-block:: python

    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
