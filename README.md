[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric
[build-image]: https://travis-ci.org/rusty1s/pytorch_geometric.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_geometric
[docs-image]: https://readthedocs.org/projects/pytorch-geometric/badge/?version=latest
[docs-url]: https://pytorch-geometric.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_geometric?branch=master
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/rusty1s/pytorch_geometric/blob/master/CONTRIBUTING.md
[slack-image]: https://img.shields.io/badge/slack-pyg-brightgreen
[slack-url]: https://join.slack.com/t/torchgeometricco/shared_invite/zt-imae9z78-U8p78pxOiYQDVGYAsQuSHg

<p align="center">
  <img height="110" src="https://raw.githubusercontent.com/rusty1s/pytorch_geometric/master/docs/source/_static/img/pyg_logo_text.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Contributing][contributing-image]][contributing-url]
[![Slack][slack-image]][slack-url]

**[Documentation](https://pytorch-geometric.readthedocs.io)** | **[Paper](https://arxiv.org/abs/1903.02428)** | **[Colab Notebooks](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)** | **[External Resources](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html)** | **[OGB Examples](https://github.com/snap-stanford/ogb/tree/master/examples)**

*PyTorch Geometric* (PyG) is a geometric deep learning extension library for [PyTorch](https://pytorch.org/).

It consists of various methods for deep learning on graphs and other irregular structures, also known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of an easy-to-use mini-batch loader for many small and single giant graphs, multi gpu-support, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.
[Click here to join our Slack community!][slack-url]

--------------------------------------------------------------------------------

PyTorch Geometric makes implementing Graph Neural Networks a breeze (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) for the accompanying tutorial).
For example, this is all it takes to implement the [edge convolutional layer](https://arxiv.org/abs/1801.07829):

```python
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, F_in, F_out):
        super(EdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = Seq(Lin(2 * F_in, F_out), ReLU(), Lin(F_out, F_out))

    def forward(self, x, edge_index):
        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)  # shape [N, F_out]

    def message(self, x_i, x_j):
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # shape [E, 2 * F_in]
        return self.mlp(edge_features)  # shape [E, F_out]
```

In detail, the following methods are currently implemented:

* **[SplineConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SplineConv)** from Fey *et al.*: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018) [[**Example1**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/cora.py), [**Example2**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/faust.py)]
* **[GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)** from Kipf and Welling: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py)]
* **[GCN2Conv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCN2Conv)** from Chen *et al.*: [Simple and Deep Graph Convolutional Networks](https://arxiv.org/abs/2007.02133) (ICML 2020) [[**Example1**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn2_cora.py), [**Example2**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn2_ppi.py)]
* **[ChebConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv)** from Defferrard *et al.*: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) (NIPS 2016) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py#L36-L37)]
* **[NNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.NNConv)** from Gilmer *et al.*: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) (ICML 2017) [[**Example1**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/qm9_nn_conv.py), [**Example2**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mnist_nn_conv.py)]
* **[CGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.CGConv)** from Xie and Grossman: [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) (Physical Review Letters 120, 2018)
* **[ECConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ECConv)** from Simonovsky and Komodakis: [Edge-Conditioned Convolution on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)
* **[GATConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)** from Veličković *et al.*: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py)]
* **[TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv)** from Shi *et al.*: [Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification](https://arxiv.org/abs/2009.03509) (CoRR 2020)
* **[SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017) [[**Example1**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py), [**Example2**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py), [**Example3**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_sage_unsup.py)]
* **[GraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)** from, *e.g.*, Morris *et al.*: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) (AAAI 2019)
* **[GatedGraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GatedGraphConv)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016)
* **[GINConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv)** from Xu *et al.*: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (ICLR 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mutag_gin.py)]
* **[GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINEConv)** from Hu *et al.*: [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) (ICLR 2020)
* **[ARMAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ARMAConv)** from Bianchi *et al.*: [Graph Neural Networks with Convolutional ARMA Filters](https://arxiv.org/abs/1901.01343) (CoRR 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/arma.py)]
* **[SGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SGConv)** from Wu *et al.*: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) (CoRR 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/sgc.py)]
* **[APPNP](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.APPNP)** from Klicpera *et al.*: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997) (ICLR 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/appnp.py)]
* **[MFConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.MFConv)** from Duvenaud *et al.*: [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292) (NIPS 2015)
* **[AGNNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.AGNNConv)** from Thekumparampil *et al.*: [Attention-based Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735) (CoRR 2017) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/agnn.py)]
* **[TAGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TAGConv)** from Du *et al.*: [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1710.10370) (CoRR 2017) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/tagcn.py)]
* **[PNAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PNAConv)** from Corso *et al.*: [Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718) (CoRR 2020) [**[Example](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py)**]
* **[RGCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv)** from Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (ESWC 2018) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/rgcn.py)]
* **[SignedConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SignedConv)** from Derr *et al.*: [Signed Graph Convolutional Network](https://arxiv.org/abs/1808.06354) (ICDM 2018) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/signed_gcn.py)]
* **[DNAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.DNAConv)** from Fey: [Just Jump: Dynamic Neighborhood Aggregation in Graph Neural Networks](https://arxiv.org/abs/1904.04849) (ICLR-W 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/dna.py)]
* **[PointConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PointConv)** (including **[Iterative Farthest Point Sampling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.fps)**, dynamic graph generation based on **[nearest neighbor](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.knn_graph)** or **[maximum distance](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.radius_graph)**, and **[k-NN interpolation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.unpool.knn_interpolate)** for upsampling) from Qi *et al.*: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) (CVPR 2017) and [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) (NIPS 2017) [[**Example1**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py), [**Example2**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py)]
* **[EdgeConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv)** from Wang *et al.*: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) (CoRR, 2018) [[**Example1**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/dgcnn_classification.py), [**Example2**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/dgcnn_segmentation.py)]
* **[XConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.XConv)** from Li *et al.*: [PointCNN: Convolution On X-Transformed Points](https://arxiv.org/abs/1801.07791) (NeurIPS 2018) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/points/point_cnn.py)]
* **[PPFConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PPFConv)** from Deng *et al.*: [PPFNet: Global Context Aware Local Features for Robust 3D Point Matching](https://arxiv.org/abs/1802.02669) (CVPR 2018)
* **[GMMConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GMMConv)** from Monti *et al.*: [Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs](https://arxiv.org/abs/1611.08402) (CVPR 2017)
* **[FeaStConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FeaStConv)** from Verma *et al.*: [FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis](https://arxiv.org/abs/1706.05206) (CVPR 2018)
* **[HypergraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HypergraphConv)** from Bai *et al.*: [Hypergraph Convolution and Hypergraph Attention](https://arxiv.org/abs/1901.08150) (CoRR 2019)
* **[GravNetConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GravNetConv)** from Qasim *et al.*: [Learning Representations of Irregular Particle-detector Geometry with Distance-weighted Graph Networks](https://arxiv.org/abs/1902.07987) (European Physics Journal C, 2019)
* A **[MetaLayer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.meta.MetaLayer)** for building any kind of graph network similar to the [TensorFlow Graph Nets library](https://github.com/deepmind/graph_nets) from Battaglia *et al.*: [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) (CoRR 2018)
* **[GlobalAttention](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.GlobalAttention)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/global_attention.py)]
* **[Set2Set](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.Set2Set)** from Vinyals *et al.*: [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391) (ICLR 2016) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/set2set.py)]
* **[Sort Pool](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.global_sort_pool)** from Zhang *et al.*: [An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) (AAAI 2018) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/sort_pool.py)]
* **[Dense Differentiable Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.diff_pool.dense_diff_pool)** from Ying *et al.*: [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) (NeurIPS 2018) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/proteins_diff_pool.py)]
* **[Dense MinCUT Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.mincut_pool.dense_mincut_pool)** from Bianchi *et al.*: [MinCUT Pooling in Graph Neural Networks](https://arxiv.org/abs/1907.00481) (CoRR 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/proteins_mincut_pool.py)]
* **[Graclus Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mnist_graclus.py)]
* **[Voxel Grid Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.voxel_grid)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mnist_voxel_grid.py)]
* **[Top-K Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.TopKPooling)** from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178) (ICML 2019), Cangea *et al.*: [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287) (NeurIPS-W 2018) and Knyazev *et al.*: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/proteins_topk_pool.py)]
* **[SAG Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.SAGPooling)** from Lee *et al.*: [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082) (ICML 2019) and Knyazev *et al.*: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/sag_pool.py)]
* **[Edge Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.EdgePooling)** from Diehl *et al.*: [Towards Graph Pooling by Edge Contraction](https://graphreason.github.io/papers/17.pdf) (ICML-W 2019) and Diehl: [Edge Contraction Pooling for Graph Neural Networks](https://arxiv.org/abs/1905.10990) (CoRR 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/edge_pool.py)]
* **[ASAPooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.ASAPooling)** from Ranjan *et al.*: [ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations](https://arxiv.org/abs/1911.07979) (AAAI 2020) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/asap.py)]
* **[Local Degree Profile](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.transforms.LocalDegreeProfile)** from Cai and Wang: [A Simple yet Effective Baseline for Non-attribute Graph Classification](https://arxiv.org/abs/1811.03508) (CoRR 2018)
* **[Jumping Knowledge](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.JumpingKnowledge)** from Xu *et al.*: [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536) (ICML 2018) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py#L54-L106)]
* **[Node2Vec](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.Node2Vec)** from Grover and Leskovec: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (KDD 2016) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/node2vec.py)]
* **[MetaPath2Vec](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.MetaPath2Vec)** from Dong *et al.*: [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) (KDD 2017) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/metapath2vec.py)]
* **[Deep Graph Infomax](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.DeepGraphInfomax)** from Veličković *et al.*: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (ICLR 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/infomax.py)]
* All variants of **[Graph Autoencoders](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GAE)** and **[Variational Autoencoders](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.VGAE)** from:
    * [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) from Kipf and Welling (NIPS-W 2016) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py)]
    * [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/abs/1802.04407) from Pan *et al.* (IJCAI 2018) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/argva_node_clustering.py)]
    * [Simple and Effective Graph Autoencoders with One-Hop Linear Models](https://arxiv.org/abs/2001.07614) from Salha *et al.* (ECML 2020) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py)]
* **[SEAL](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/seal_link_pred.py)** from Zhang and Chen: [Link Prediction Based on Graph Neural Networks](https://arxiv.org/pdf/1802.09691.pdf) (NeurIPS 2018)
* **[RENet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.RENet)** from Jin *et al.*: [Recurrent Event Network for Reasoning over Temporal Knowledge Graphs](https://arxiv.org/abs/1904.05530) (ICLR-W 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/renet.py)]
* **[GraphUNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GraphUNet)** from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178) (ICML 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_unet.py)]
* **[SchNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.SchNet)** from Schütt *et al.*: [SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions](https://arxiv.org/abs/1706.08566) (NIPS 2017) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/qm9_pretrained_schnet.py)]
* **[DimeNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.DimeNet)** from Klicpera *et al.*: [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123) (ICLR 2020) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/qm9_pretrained_dimenet.py)]
* **[DeepGCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.DeepGCNLayer)** and the **[GENConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GENConv)** from Li *et al.*: [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://arxiv.org/abs/1904.03751) (ICCV 2019) and [DeeperGCN: All You Need to Train Deeper GCNs](https://arxiv.org/abs/2006.07739) (CoRR 2020) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py)]
* **[NeighborSampler](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.NeighborSampler)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017) [[**Example1**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py), [**Example2**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py), [**Example3**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_gat.py)]
* **[ClusterGCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.ClusterLoader)** from Chiang *et al.*: [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953) (KDD 2019) [[**Example1**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py), [**Example2**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/cluster_gcn_ppi.py)]
* **[GraphSAINT](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.GraphSAINTSampler)** from Zeng *et al.*: [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931) (ICLR 2020) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py)]
* **[GDC](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GDC)** from Klicpera *et al.*: [Diffusion Improves Graph Learning](https://arxiv.org/abs/1911.05485) (NeurIPS 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py)]
* **[SIGN](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.SIGN)** from Rossi *et al.*: [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198) (CoRR 2020) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/sign.py)]
* **[GraphSizeNorm](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.GraphSizeNorm)** from Dwivedi *et al.*: [Benchmarking Graph Neural Networks](https://arxiv.org/abs/2003.00982) (CoRR 2020)
* **[GNNExplainer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GNNExplainer)** from Ying *et al.*: [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) (NeurIPS 2019) [[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gnn_explainer.py)]
* **[DropEdge](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_adj)** from Rong *et al.*: [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://openreview.net/forum?id=Hkx1qkrKPr) (ICLR 2020)
* **[PairNorm](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.PairNorm)** from Zhao and Akoglu: [PairNorm: Tackling Oversmoothing in GNNs](https://arxiv.org/abs/1909.12223) (ICLR 2020)
* **[Tree Decomposition](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.tree_decomposition)** from Jin *et al.*: [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/abs/1802.04364) (ICML 2018)

--------------------------------------------------------------------------------

Head over to our [documentation](https://pytorch-geometric.readthedocs.io) to find out more about installation, data handling, creation of datasets and a full list of implemented methods, transforms, and datasets.
For a quick start, check out our [examples](https://github.com/rusty1s/pytorch_geometric/tree/master/examples) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](https://github.com/rusty1s/pytorch_geometric/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/rusty1s/pytorch_geometric/issues).
We are motivated to constantly make PyTorch Geometric even better.

## Installation

We provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://pytorch-geometric.com/whl).

### PyTorch 1.7.0

To install the binaries for PyTorch 1.7.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, or `cu110` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` | `cu110` |
|-------------|-------|--------|---------|---------|---------|
| **Linux**   | ✅    | ✅     | ✅      | ✅      | ✅      |
| **Windows** | ✅    | ❌     | ✅      | ✅      | ✅      |
| **macOS**   | ✅    |        |         |         |         |

### PyTorch 1.6.0

To install the binaries for PyTorch 1.6.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, or `cu102` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` |
|-------------|-------|--------|---------|---------|
| **Linux**   | ✅    | ✅     | ✅      | ✅      |
| **Windows** | ✅    | ❌     | ✅      | ✅      |
| **macOS**   | ✅    |        |         |         |

**Note:** Binaries of older versions are also provided for PyTorch 1.4.0 and PyTorch 1.5.0/1.5.1 (following the same procedure).

### From master

In case you want to experiment with the latest PyG features which did not get released yet, you can install PyG from master via

```
pip install git+https://github.com/rusty1s/pytorch_geometric.git
```

## Running examples

```
cd examples
python gcn.py
```

## Cite

Please cite [our paper](https://arxiv.org/abs/1903.02428) (and the respective papers of the methods used) if you use this code in your own work:

```
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```

Feel free to [email us](mailto:matthias.fey@tu-dortmund.de) if you wish your work to be listed in the [external resources](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html).

## Running tests

```
$ python setup.py test
```
