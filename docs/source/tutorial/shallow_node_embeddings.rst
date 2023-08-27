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

Here, I'll write something about Node2Vec.


MetaPath2Vec
------------

Here, I'll write something about MetaPath2Vec.
