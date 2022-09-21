:github_url: https://github.com/pyg-team/pytorch_geometric

PyG Documentation
=================

**PyG** *(PyTorch Geometric)* is a library built upon `PyTorch <https://pytorch.org/>`_ to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.

It consists of various methods for deep learning on graphs and other irregular structures, also known as `geometric deep learning <http://geometricdeeplearning.com/>`_, from a variety of published papers.
In addition, it consists of easy-to-use mini-batch loaders for operating on many small and single giant graphs, `multi GPU-support <https://github.com/pyg-team/pytorch_geometric/tree/master/examples/multi_gpu>`_, `DataPipe support <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/datapipe.py>`_, distributed graph learning via `Quiver <https://github.com/pyg-team/pytorch_geometric/tree/master/examples/quiver>`_, a large number of common benchmark datasets (based on simple interfaces to create your own), the `GraphGym <https://pytorch-geometric.readthedocs.io/en/latest/notes/graphgym.html>`_ experiment manager, and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.
`Click here to join our Slack community! <https://data.pyg.org/slack.html>`_

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/introduction
   notes/create_gnn
   notes/create_dataset
   notes/heterogeneous
   notes/load_csv
   notes/graphgym
   notes/batching
   notes/sparse_tensor
   notes/jit
   notes/cheatsheet
   notes/data_cheatsheet
   notes/colabs
   notes/resources

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/root
   modules/nn
   modules/data
   modules/loader
   modules/datasets
   modules/transforms
   modules/utils
   modules/graphgym
   modules/profile

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
