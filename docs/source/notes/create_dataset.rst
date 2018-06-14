Create your own dataset
=======================

Although `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ already contains a lot of useful datasets, you may wish to create your own because you recorded data by yourself or want to use a dataset which is not publicly available.

Implementing datasets by yourself is straightforward and you may have a look at the source code to find out how the various datasets are implemented.
However, we give a brief introduction on what is needed to setup your own dataset.

We provide two abstract classes for datasets: :class:`torch_geometric.data.Dataset` and :class:`torch_geometric.data.InMemoryDataset`.
:class:`torch_geometric.data.InMemoryDataset` inherits from :class:`torch_geometric.data.Dataset` and should be used if the whole dataset fits into memory.

Each dataset gets passed a root folder, a ``transform`` and a ``pre_transform`` object (``None`` by default).
We basically split up the root folder into two folders: the ``raw_dir``, where the dataset gets downloaded to, and the ``processed_dir``, where the processed dataset is being saved.

Creating "in memory datasets"
-----------------------------

In order to create a :class:`torch_geometric.data.InMemoryDataset`, you need to implement four fundamental methods:

:meth:`torch_geometric.data.InMemoryDataset.raw_file_names`:
    A list of files in the ``raw_dir`` which needs to be found, so the download is being skipped.
:meth:`torch_geometric.data.InMemoryDataset.processed_file_names`:
    A list of files in the ``processed_dir`` which needs to be found, so the processing is being skipped.
:meth:`torch_geometric.data.InMemoryDataset.download`:
    Should download all needed data into ``raw_dir``.
:meth:`torch_geometric.data.InMemoryDataset.process`:
    Should process downloaded data and save it into the ``processed_dir``.

You can find helpful methods to download and extract data in :mod:`torch_geometric.data`.

The real magic happens in the body of :meth:`torch_geometric.data.InMemoryDataset.process`.
Here, we need to read and create a list of :class:`torch_geometric.data.Data` objects and save it into the ``processed_dir``.
Because saving a huge python list is really slow, we must collate the list into one huge ``Data`` object before saving.
The collated data object simply concatenates all examples (similar to :class:`torch_geometric.data.DataLoader`).
In addition, the :meth:`torch_geometric.data.InMemoryDataset.collate` method returns a ``slices`` dictionary to reconstruct single examples from the object.

Let's see this process in a simplified example:

.. code-block:: python

    import torch
    from torch_geometric.data import InMemoryDataset


    class MyOwnDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(MyOwnDataset, self).__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return ['some_file_1', 'some_file_2', ...]

        @property
        def processed_file_names(self):
            return ['data.pt']

        def download(self):
            # Download to `self.raw_dir`

        def process(self):
            # Read data into huge `Data` list.
            data_list = [...]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

Creating "larger" datasets
--------------------------

Analogous to the datasets in ``torchvision``, you need to further implement the following methods:

:meth:`torch_geometric.data.Dataset.__len__`:
    How many graphs are in your dataset?

:meth:`torch_geometric.data.Dataset.get`:
    Logic to load a single graph.
    The ``Data`` object will automatically be transformed according to ``self.transform``.

In :meth:`torch_geometric.data.Dataset.process`, each graph data object should be saved individually and be manually loaded in :meth:`torch_geometric.data.Dataset.get`.
