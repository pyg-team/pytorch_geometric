:github_url: https://github.com/pyg-team/pytorch_geometric

PyG Documentation
=================

:pyg:`null` **PyG** *(PyTorch Geometric)* is a library built upon :pytorch:`null` `PyTorch <https://pytorch.org>`_ to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.

It consists of various methods for deep learning on graphs and other irregular structures, also known as `geometric deep learning <http://geometricdeeplearning.com/>`_, from a variety of published papers.
In addition, it consists of easy-to-use mini-batch loaders for operating on many small and single giant graphs, `multi GPU-support <https://github.com/pyg-team/pytorch_geometric/tree/master/examples/multi_gpu>`_, `torch.compile <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/compile.html>`_ support, `DataPipe <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/datapipe.py>`_ support, a large number of common benchmark datasets (based on simple interfaces to create your own), the `GraphGym <advanced/graphgym.html>`__ experiment manager, and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

.. slack_button::

.. toctree::
   :maxdepth: 1
   :caption: Install PyG

   install/installation

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/introduction
   get_started/colabs

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/create_gnn
   tutorial/create_dataset
   tutorial/heterogeneous
   tutorial/load_csv
   tutorial/explain
   tutorial/compile

.. toctree::
   :maxdepth: 1
   :caption: Advanced Concepts

   advanced/batching
   advanced/sparse_tensor
   advanced/hgam
   advanced/jit
   advanced/remote
   advanced/graphgym
   advanced/cpu_affinity

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   modules/root
   modules/nn
   modules/data
   modules/loader
   modules/sampler
   modules/datasets
   modules/transforms
   modules/utils
   modules/explain
   modules/contrib
   modules/graphgym
   modules/profile

.. toctree::
   :maxdepth: 1
   :caption: Cheatsheets

   cheatsheet/gnn_cheatsheet
   cheatsheet/data_cheatsheet

.. toctree::
   :maxdepth: 1
   :caption: External Resources

   external/resources
