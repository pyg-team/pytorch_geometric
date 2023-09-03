Shallow Node Embeddings
=========================

In this tutorial, we will look at learning unsupervised *shallow* node embeddings using PyG for both homogenous and heterogenous graphs.

Introduction to Shallow Node Embeddings
---------------------------------------

The key difference between *shallow* node embeddings (e.g. node2vec) and *deep* node embeddings (e.g. GNNs) is the choice of encoder. For shallow node embeddings, we model the encoder as an embedding matrix look-up as follows:

.. math::
    \textrm{ENC}(u) = \mathbf{Z}[u], \quad \mathbf{Z} \in \mathbb{R}^{|\mathcal{V}| \times d}

where :math:`|\mathcal{V}|` denotes the number of nodes in the graph and :math:`d` denotes the dimension of the embeddings. Given two node embeddings :math:`\mathbf{z}_u` and :math:`\mathbf{z}_v`, we often want to reconstruct some graph-based similarity measure :math:`\mathbf{S}[u,v]` using the dot product of the embeddings:

.. math::
    \mathbf{z}_u^\top \mathbf{z}_v \approx \mathbf{S}[u,v]

There are many ways of defining the node similarity measure :math:`\mathbf{S}[u,v]`, each of them giving rise to a different set of embeddings. In this tutorial, we will look at two different methods for learning node embeddings, Node2Vec and MetaPath2Vec.

Node2Vec
--------

.. note::

   In this section of the tutorial, we will learn node embeddings for **homogenous graphs** using the :class:`~torch_geometric.nn.Node2Vec` module of :pyg:`PyG`.
   The code is available as `examples/node2vec.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py>`_ and as a `Google Colab tutorial notebook <https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial11/Tutorial11.ipynb>`_.
   For the original paper, see `node2vec: Scalable Feature Learning for Networks <https://arxiv.org/pdf/1607.00653.pdf>`_.

Node2Vec is a method for learning shallow node embeddings for homogenous graphs, where the goal is to learn embeddings such that the following holds:

.. math::
    \begin{align}
    \textrm{DEC}(\mathbf{z}_u, \mathbf{z}_v) &\triangleq \frac{\textrm{e}^{\mathbf{z}_u^\top \mathbf{z}_v}}{\sum_{v_k \in \mathcal{V}} \textrm{e}^{\mathbf{z}_u^\top \mathbf{z}_k}} \\
                                             &\approx p_{\mathcal{G}, T}(v|u)
    \end{align}

where :math:`p_{\mathcal{G},T}` is the probability of visiting node :math:`v` on a length-:math:`T` random walk starting at node :math:`u`.
Since we want to predict a probability, we use the softmax function to make sure the output is in the range :math:`[0,1]`.

To begin the example, let us load in the needed packages and the data that we will be working with.

.. code-block:: python

    import os.path as osp
    import sys
    import torch
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import Node2Vec
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    dataset = 'Cora'
    path = osp.join('data', dataset)
    dataset = Planetoid(path, dataset)
    data = dataset[0]

From inspecting the data, we can see that we have 2708 nodes and 10556 edges in the Cora dataset as follows:

.. code-block:: python

    data.num_nodes
    >>> 2708

    data.num_edges
    >>> 10556

Let us now initialize the :class:`~torch_geometric.nn.Node2Vec` module.

.. code-block:: python

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

    num_workers = 2
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

The :class:`~torch_geometric.nn.Node2Vec` module contains the learned embeddings, which are initialized from a :math:`\mathcal{N}(0,1)` distribution. The underlying :math:`N \times d`-tensor can be accessed e.g. via
:obj:`model.embedding.weight`. To generate random walks, we can simply iterate over the dataloader.

.. code-block:: python

    pos_rw, neg_rw = next(iter(loader))

Here, :obj:`pos_rw` will be the sampled positive random walks (i.e. random walks over the graph) and :obj:`neg_rw` will essentially sampled using :obj:`torch.random.randint`.
Using this dataloader and the built in :obj:`model.loss`, we can define our :obj:`train` function.

.. code-block:: python

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

Since we want to evaluate our learned embeddings on a downstream classification task, we use the :obj:`model.test` method, which fits a :obj:`LogisticRegression` model from sklearn.

.. code-block:: python

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask],
                        max_iter=150)
        return acc

Having defined the :obj:`train` and :obj:`test` function, we now train a model for 100 epochs and evaluate the accuracy. It should be noted that below, we are learning the
embeddings in an unsupervised manner. Furthermore, the node features are not used when learning the embeddings.

.. code-block:: python

    for epoch in range(1, 101):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

After running the above code, the final accuracy is approximately 70\% for the Cora classification task.

MetaPath2Vec
------------

.. note::

   In this section of the tutorial, we will learn node embeddings for **heterogenous graphs** using the :class:`~torch_geometric.nn.MetaPath2Vec` module of :pyg:`PyG`.
   The code is available as `examples/hetero/metapath2vec.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/metapath2vec.py>`_ and as a `Google Colab tutorial notebook <https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial11/Tutorial11.ipynb>`_.
   For the original paper, see `metapath2vec: Scalable Representation Learning for Heterogeneous Networks <https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf>`_.


Here, I will write some stuff about MetaPath2Vec.
