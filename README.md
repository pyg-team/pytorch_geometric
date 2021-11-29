[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric
[testing-image]: https://github.com/pyg-team/pytorch_geometric/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pyg-team/pytorch_geometric/actions/workflows/testing.yml
[linting-image]: https://github.com/pyg-team/pytorch_geometric/actions/workflows/linting.yml/badge.svg
[linting-url]: https://github.com/pyg-team/pytorch_geometric/actions/workflows/linting.yml
[docs-image]: https://readthedocs.org/projects/pytorch-geometric/badge/?version=latest
[docs-url]: https://pytorch-geometric.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/pyg-team/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/pyg-team/pytorch_geometric?branch=master
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/pyg-team/pytorch_geometric/blob/master/CONTRIBUTING.md
[slack-image]: https://img.shields.io/badge/slack-pyg-brightgreen
[slack-url]: https://join.slack.com/t/torchgeometricco/shared_invite/zt-p6br3yuo-BxRoe36OHHLF6jYU8xHtBA

<p align="center">
  <img height="150" src="https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_static/img/pyg1.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Testing Status][testing-image]][testing-url]
[![Linting Status][linting-image]][linting-url]
[![Docs Status][docs-image]][docs-url]
[![Contributing][contributing-image]][contributing-url]
[![Slack][slack-image]][slack-url]

**[Documentation](https://pytorch-geometric.readthedocs.io)** | **[Paper](https://arxiv.org/abs/1903.02428)** | **[Colab Notebooks and Video Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)** | **[External Resources](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html)** | **[OGB Examples](https://github.com/snap-stanford/ogb/tree/master/examples)**

**PyG** *(PyTorch Geometric)* is a library built upon [PyTorch](https://pytorch.org/) to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.

It consists of various methods for deep learning on graphs and other irregular structures, also known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of easy-to-use mini-batch loaders for operating on many small and single giant graphs, [multi GPU-support](https://github.com/pyg-team/pytorch_geometric/tree/master/examples/multi_gpu), distributed graph learning via [Quiver](https://github.com/quiver-team/torch-quiver), a large number of common benchmark datasets (based on simple interfaces to create your own), the [GraphGym](https://pytorch-geometric.readthedocs.io/en/latest/notes/graphgym.html) experiment manager, and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.
[Click here to join our Slack community!][slack-url]

--------------------------------------------------------------------------------

* [Library Highlights](#library-highlights)
* [Quick Tour for New Users](#quick-tour-for-new-users)
* [Architecture Overview](#architecture-overview)
* [Implemented GNN Models](#implemented-gnn-models)
* [Installation](#installation)

## Library Highlights

Whether you are a machine learning researcher or first-time user of machine learning toolkits, here are some reasons to try out PyG for machine learning on graph-structured data.

* **Easy-to-use and unified API**:
  All it takes is 10-20 lines of code to get started with training a GNN model (see the next section for a [quick tour](#quick-tour-for-new-users)).
  PyG is *PyTorch-on-the-rocks*: It utilizes a tensor-centric API and keeps design principles close to vanilla PyTorch.
  If you are already familiar with PyTorch, utilizing PyG is straightforward.
* **Comprehensive and well-maintained GNN models**:
  Most of the state-of-the-art Graph Neural Network architectures have been implemented by library developers or authors of research papers and are ready to be applied.
* **Great flexibility**:
  Existing PyG models can easily be extended for conducting your own research with GNNs.
  Making modifications to existing models or creating new architectures is simple, thanks to its easy-to-use message passing API, and a variety of operators and utility functions.
* **Large-scale real-world GNN models**:
  We focus on the need of GNN applications in challenging real-world scenarios, and support learning on diverse types of graphs, including but not limited to: scalable GNNs for graphs with millions of nodes; dynamic GNNs for node predictions over time; heterogeneous GNNs with multiple node types and edge types.
* **GraphGym integration**: GraphGym lets users easily reproduce GNN experiments, is able to launch and analyze thousands of different GNN configurations, and is customizable by registering new modules to a GNN learning pipeline.

## Quick Tour for New Users

In this quick tour, we highlight the ease of creating and training a GNN model with only a few lines of code.

### Train your own GNN model

In the first glimpse of PyG, we implement the training of a GNN for classifying papers in a citation graph.
For this, we load the [Cora](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid) dataset, and create a simple 2-layer GCN model using the pre-defined [`GCNConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv):

```python
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='.', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, 16, dataset.num_classes)
```

<details>
<summary>We can now optimize the model in a training loop, similar to the <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation">standard PyTorch training procedure</a>.</summary>

```python
import torch.nn.functional as F

data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    pred = model(data.x, data.edge_index)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
</details>

More information about evaluating final model performance can be found in the corresponding [example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py).

### Create your own GNN layer

In addition to the easy application of existing GNNs, PyG makes it simple to implement custom Graph Neural Networks (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) for the accompanying tutorial).
For example, this is all it takes to implement the [edge convolutional layer](https://arxiv.org/abs/1801.07829) from Wang *et al.*:

<p align="center">
  <img height="40" src="https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_figures/edge_conv.svg?sanitize=true" />
</p>

```python
import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")  # "Max" aggregation.
        self.mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        return self.propagate(edge_index, x=x)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)  # shape [num_edges, out_channels]
```

### Manage experiments with GraphGym

GraphGym allows you to manage and launch GNN experiments, using a highly modularized pipeline (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/graphgym.html) for the accompanying tutorial).

```
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric/graphgym
bash run_single.sh  # run a single GNN experiment (node/edge/graph-level)
bash run_batch.sh   # run a batch of GNN experiments, using differnt GNN designs/datasets/tasks
```

Users are highly encouraged to check out the [documentation](https://pytorch-geometric.readthedocs.io/en/latest), which contains additional tutorials on the essential functionalities of PyG, including data handling, creation of datasets and a full list of implemented methods, transforms, and datasets.
For a quick start, check out our [examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples) in `examples/`.

## Architecture Overview

PyG provides a multi-layer framework that enables users to build Graph Neural Network solutions on both low and high levels.
It comprises of the following components:

* The PyG **engine** utilizes the powerful PyTorch deep learning framework, as well as additions of efficient CUDA libraries for operating on sparse data, *e.g.*, [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse) and [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster).
* The PyG **storage** handles data processing, transformation and loading pipelines. It is capable of handling and processing large-scale graph datasets, and provides effective solutions for heterogeneous graphs. It further provides a variety of sampling solutions, which enable training of GNNs on large-scale graphs.
* The PyG **operators** bundle essential functionalities for implementing Graph Neural Networks. PyG supports important GNN building blocks that can be combined and applied to various parts of a GNN model, ensuring rich flexibility of GNN design.
* Finally, PyG provides an abundant set of GNN **models**, and examples that showcase GNN models on standard graph benchmarks. Thanks to its flexibility, users can easily build and modify custom GNN models to fit their specific needs.

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_static/img/architecture.svg?sanitize=true" />
</p>

## Implemented GNN Models

We list currently supported PyG models, layers and operators according to category:

**GNN layers:**
All Graph Neural Network layers are implemented via the **[`nn.MessagePassing`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing)** interface.
A GNN layer specifies how to perform message passing, *i.e.* by designing different message, aggregation and update functions as defined [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html).
These GNN layers can be stacked together to create Graph Neural Network models.

* **[GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)** from Kipf and Welling: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py)]
* **[ChebConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv)** from Defferrard *et al.*: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) (NIPS 2016) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py#L36-L37)]
* **[GATConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)** from Veličković *et al.*: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py)]

<details>
<summary><b>Expand to see all implemented GNN layers...</b></summary>

* **[GCN2Conv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCN2Conv)** from Chen *et al.*: [Simple and Deep Graph Convolutional Networks](https://arxiv.org/abs/2007.02133) (ICML 2020) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn2_cora.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn2_ppi.py)]
* **[SplineConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SplineConv)** from Fey *et al.*: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cora.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/faust.py)]
* **[NNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.NNConv)** from Gilmer *et al.*: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) (ICML 2017) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_nn_conv.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_nn_conv.py)]
* **[CGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.CGConv)** from Xie and Grossman: [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) (Physical Review Letters 120, 2018)
* **[ECConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ECConv)** from Simonovsky and Komodakis: [Edge-Conditioned Convolution on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)
* **[EGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EGConv)** from Tailor *et al.*: [Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions](https://arxiv.org/abs/2104.01481) (GNNSys 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/egc.py)]
* **[GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv)** from Brody *et al.*: [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) (CoRR 2021)
* **[TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv)** from Shi *et al.*: [Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification](https://arxiv.org/abs/2009.03509) (CoRR 2020)
* **[SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py), [**Example3**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup.py)]
* **[GraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)** from, *e.g.*, Morris *et al.*: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) (AAAI 2019)
* **[GatedGraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GatedGraphConv)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016)
* **[ResGatedGraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ResGatedGraphConv)** from Bresson and Laurent: [Residual Gated Graph ConvNets](https://arxiv.org/abs/1711.07553) (CoRR 2017)
* **[GINConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv)** from Xu *et al.*: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (ICLR 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mutag_gin.py)]
* **[GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINEConv)** from Hu *et al.*: [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) (ICLR 2020)
* **[ARMAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ARMAConv)** from Bianchi *et al.*: [Graph Neural Networks with Convolutional ARMA Filters](https://arxiv.org/abs/1901.01343) (CoRR 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/arma.py)]
* **[SGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SGConv)** from Wu *et al.*: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) (CoRR 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sgc.py)]
* **[APPNP](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.APPNP)** from Klicpera *et al.*: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997) (ICLR 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/citation/appnp.py)]
* **[MFConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.MFConv)** from Duvenaud *et al.*: [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292) (NIPS 2015)
* **[AGNNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.AGNNConv)** from Thekumparampil *et al.*: [Attention-based Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735) (CoRR 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/agnn.py)]
* **[TAGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TAGConv)** from Du *et al.*: [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1710.10370) (CoRR 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tagcn.py)]
* **[PNAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PNAConv)** from Corso *et al.*: [Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718) (CoRR 2020) [**[Example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py)**]
* **[FAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FAConv)** from Bo *et al.*: [Beyond Low-Frequency Information in Graph Convolutional Networks](https://arxiv.org/abs/2101.00797) (AAAI 2021)
* **[PDNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.nn.conv.PDNConv)** from Rozemberczki *et al.*: [Pathfinder Discovery Networks for Neural Message Passing](https://arxiv.org/abs/2010.12878) (WWW 2021)
* **[RGCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv)** from Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (ESWC 2018) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn_link_pred.py)]
* **[FiLMConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FiLMConv)** from Brockschmidt: [GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation](https://arxiv.org/abs/1906.12192) (ICML 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/film.py)]
* **[SignedConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SignedConv)** from Derr *et al.*: [Signed Graph Convolutional Network](https://arxiv.org/abs/1808.06354) (ICDM 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/signed_gcn.py)]
* **[DNAConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.DNAConv)** from Fey: [Just Jump: Dynamic Neighborhood Aggregation in Graph Neural Networks](https://arxiv.org/abs/1904.04849) (ICLR-W 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dna.py)]
* **[PANConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PANConv)** from Ma *et al.*: [Path Integral Based Convolution and Pooling for Graph Neural Networks](https://arxiv.org/abs/2006.16811) (NeurIPS 2020)
* **[PointConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PointConv)** (including **[Iterative Farthest Point Sampling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.fps)**, dynamic graph generation based on **[nearest neighbor](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.knn_graph)** or **[maximum distance](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.radius_graph)**, and **[k-NN interpolation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.unpool.knn_interpolate)** for upsampling) from Qi *et al.*: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) (CVPR 2017) and [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) (NIPS 2017) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py)]
* **[EdgeConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv)** from Wang *et al.*: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) (CoRR, 2018) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_classification.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_segmentation.py)]
* **[XConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.XConv)** from Li *et al.*: [PointCNN: Convolution On X-Transformed Points](https://arxiv.org/abs/1801.07791) (NeurIPS 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/points/point_cnn.py)]
* **[PPFConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PPFConv)** from Deng *et al.*: [PPFNet: Global Context Aware Local Features for Robust 3D Point Matching](https://arxiv.org/abs/1802.02669) (CVPR 2018)
* **[GMMConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GMMConv)** from Monti *et al.*: [Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs](https://arxiv.org/abs/1611.08402) (CVPR 2017)
* **[FeaStConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FeaStConv)** from Verma *et al.*: [FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis](https://arxiv.org/abs/1706.05206) (CVPR 2018)
* **[PointTransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PointTransformerConv)** from Zhao *et al.*: [Point Transformer](https://arxiv.org/abs/2012.09164) (2020)
* **[HypergraphConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HypergraphConv)** from Bai *et al.*: [Hypergraph Convolution and Hypergraph Attention](https://arxiv.org/abs/1901.08150) (CoRR 2019)
* **[GravNetConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GravNetConv)** from Qasim *et al.*: [Learning Representations of Irregular Particle-detector Geometry with Distance-weighted Graph Networks](https://arxiv.org/abs/1902.07987) (European Physics Journal C, 2019)
* **[SuperGAT](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SuperGATConv)** from Kim and Oh: [How To Find Your Friendly Neighborhood: Graph Attention Design With Self-Supervision](https://openreview.net/forum?id=Wi5KUNlqWty) (ICLR 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/super_gat.py)]
* **[HGTConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HGTConv)** from Hu *et al.*: [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) (WWW 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py)]
* **[HEATConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HEATonv)** from Mo *et al.*: [Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent Trajectory Prediction](https://arxiv.org/abs/2106.07161) (CoRR 2021)
* A **[MetaLayer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.meta.MetaLayer)** for building any kind of graph network similar to the [TensorFlow Graph Nets library](https://github.com/deepmind/graph_nets) from Battaglia *et al.*: [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) (CoRR 2018)
</details>

**Pooling layers:**
Graph pooling layers combine the vectorial representations of a set of nodes in a graph (or a subgraph) into a single vector representation that summarizes its properties of nodes.
It is commonly applied to graph-level tasks, which require combining node features into a single graph representation.

* **[Top-K Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.TopKPooling)** from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178) (ICML 2019), Cangea *et al.*: [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287) (NeurIPS-W 2018) and Knyazev *et al.*: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_topk_pool.py)]
* **[DiffPool](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.diff_pool.dense_diff_pool)** from Ying *et al.*: [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) (NeurIPS 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py)]

<details>
<summary><b>Expand to see all implemented pooling layers...</b></summary>

* **[GlobalAttention](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.GlobalAttention)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/global_attention.py)]
* **[Set2Set](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.Set2Set)** from Vinyals *et al.*: [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391) (ICLR 2016) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/set2set.py)]
* **[Sort Pool](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.glob.global_sort_pool)** from Zhang *et al.*: [An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) (AAAI 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/sort_pool.py)]
* **[Dense MinCUT Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.mincut_pool.dense_mincut_pool)** from Bianchi *et al.*: [MinCUT Pooling in Graph Neural Networks](https://arxiv.org/abs/1907.00481) (CoRR 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_mincut_pool.py)]
* **[Graclus Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_graclus.py)]
* **[Voxel Grid Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.voxel_grid)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_voxel_grid.py)]
* **[SAG Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.SAGPooling)** from Lee *et al.*: [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082) (ICML 2019) and Knyazev *et al.*: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/sag_pool.py)]
* **[Edge Pooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.EdgePooling)** from Diehl *et al.*: [Towards Graph Pooling by Edge Contraction](https://graphreason.github.io/papers/17.pdf) (ICML-W 2019) and Diehl: [Edge Contraction Pooling for Graph Neural Networks](https://arxiv.org/abs/1905.10990) (CoRR 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/edge_pool.py)]
* **[ASAPooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.ASAPooling)** from Ranjan *et al.*: [ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations](https://arxiv.org/abs/1911.07979) (AAAI 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/asap.py)]
* **[PANPooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.PANPooling)** from Ma *et al.*: [Path Integral Based Convolution and Pooling for Graph Neural Networks](https://arxiv.org/abs/2006.16811) (NeurIPS 2020)
* **[MemPooling](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.MemPooling)** from Khasahmadi *et al.*: [Memory-Based Graph Networks](https://arxiv.org/abs/2002.09518) (ICLR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mem_pool.py)]
* **[GraphMultisetTransformer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.GraphMultisetTransformer)** from Baek *et al.*: [Accurate Learning of Graph Representations with Graph Multiset Pooling](https://arxiv.org/abs/2102.11533) (ICLR 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_gmt.py)]
</details>

**GNN models:**
Our supported GNN models incorporate multiple message passing layers, and users can directly use these pre-defined models to make predictions on graphs.
Unlike simple stacking of GNN layers, these models could involve pre-processing, additional learnable parameters, skip connections, graph coarsening, etc.

* **[SchNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.SchNet)** from Schütt *et al.*: [SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions](https://arxiv.org/abs/1706.08566) (NIPS 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_pretrained_schnet.py)]
* **[DimeNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.DimeNet)** from Klicpera *et al.*: [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123) (ICLR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_pretrained_dimenet.py)]
* **[Node2Vec](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.Node2Vec)** from Grover and Leskovec: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (KDD 2016) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py)]
* **[Deep Graph Infomax](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.DeepGraphInfomax)** from Veličković *et al.*: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (ICLR 2019) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_transductive.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_inductive.py)]

<details>
<summary><b>Expand to see all implemented GNN models...</b></summary>

* **[Jumping Knowledge](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.JumpingKnowledge)** from Xu *et al.*: [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536) (ICML 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gin.py#L54-L106)]
* **[MetaPath2Vec](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.MetaPath2Vec)** from Dong *et al.*: [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) (KDD 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/metapath2vec.py)]
* All variants of **[Graph Autoencoders](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GAE)** and **[Variational Autoencoders](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.VGAE)** from:
    * [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) from Kipf and Welling (NIPS-W 2016) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py)]
    * [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/abs/1802.04407) from Pan *et al.* (IJCAI 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py)]
    * [Simple and Effective Graph Autoencoders with One-Hop Linear Models](https://arxiv.org/abs/2001.07614) from Salha *et al.* (ECML 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py)]
* **[SEAL](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py)** from Zhang and Chen: [Link Prediction Based on Graph Neural Networks](https://arxiv.org/pdf/1802.09691.pdf) (NeurIPS 2018)
* **[RENet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.RENet)** from Jin *et al.*: [Recurrent Event Network for Reasoning over Temporal Knowledge Graphs](https://arxiv.org/abs/1904.05530) (ICLR-W 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/renet.py)]
* **[GraphUNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GraphUNet)** from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178) (ICML 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_unet.py)]
* **[AttentiveFP](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.AttentiveFP)** from Xiong *et al.*: [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959) (J. Med. Chem. 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py)]
* **[DeepGCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.DeepGCNLayer)** and the **[GENConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GENConv)** from Li *et al.*: [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://arxiv.org/abs/1904.03751) (ICCV 2019) and [DeeperGCN: All You Need to Train Deeper GCNs](https://arxiv.org/abs/2006.07739) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py)]
* **[RECT](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.RECT_L)** from Wang *et al.*: [Network Embedding with Completely-imbalanced Labels](https://ieeexplore.ieee.org/document/8979355) (TKDE 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rect.py)]
* **[GNNExplainer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GNNExplainer)** from Ying *et al.*: [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) (NeurIPS 2019) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gnn_explainer.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gnn_explainer_ba_shapes.py)]
* **SEAL** from Zhang and Chen: [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691) (NeurIPS 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py)]
* **Graph-less Neural Networks** from Zhang *et al.*: [Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation](https://arxiv.org/abs/2110.08727) (CoRR 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/glnn.py)]
</details>

**GNN operators and utilities:**
PyG comes with a rich set of neural network operators that are commonly used in many GNN models.
They follow an extensible design: It is easy to apply these operators and graph utilities to existing GNN layers and models to further enhance model performance.

* **[DropEdge](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_adj)** from Rong *et al.*: [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://openreview.net/forum?id=Hkx1qkrKPr) (ICLR 2020)
* **[GraphNorm](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.GraphNorm)** from Cai *et al.*: [GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training](https://proceedings.mlr.press/v139/cai21e.html) (ICML 2021)
* **[GDC](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GDC)** from Klicpera *et al.*: [Diffusion Improves Graph Learning](https://arxiv.org/abs/1911.05485) (NeurIPS 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py)]

<details>
<summary><b>Expand to see all implemented GNN operators and utilities...</b></summary>

* **[GraphSizeNorm](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.GraphSizeNorm)** from Dwivedi *et al.*: [Benchmarking Graph Neural Networks](https://arxiv.org/abs/2003.00982) (CoRR 2020)
* **[PairNorm](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.PairNorm)** from Zhao and Akoglu: [PairNorm: Tackling Oversmoothing in GNNs](https://arxiv.org/abs/1909.12223) (ICLR 2020)
* **[DiffGroupNorm](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.DiffGroupNorm)** from Zhou *et al.*: [Towards Deeper Graph Neural Networks with Differentiable Group Normalization](https://arxiv.org/abs/2006.06972) (NeurIPS 2020)
* **[Tree Decomposition](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.tree_decomposition)** from Jin *et al.*: [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/abs/1802.04364) (ICML 2018)
* **[TGN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.TGNMemory)** from Rossi *et al.*: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637) (GRL+ 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py)]
* **[Weisfeiler Lehman Algorithm](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.WLConv)** from Weisfeiler and Lehman: [A Reduction of a Graph to a Canonical Form and an Algebra Arising During this Reduction](https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf) (Nauchno-Technicheskaya Informatsia 1968) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/wl_kernel.py)]
* **[Label Propagation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.LabelPropagation)** from Zhu and Ghahramani: [Learning from Labeled and Unlabeled Data with Label Propagation](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf) (CMU-CALD 2002) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/label_prop.py)]
* **[Local Degree Profile](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.transforms.LocalDegreeProfile)** from Cai and Wang: [A Simple yet Effective Baseline for Non-attribute Graph Classification](https://arxiv.org/abs/1811.03508) (CoRR 2018)
* **[CorrectAndSmooth](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.CorrectAndSmooth)** from Huang *et al.*: [Combining Label Propagation And Simple Models Out-performs Graph Neural Networks](https://arxiv.org/abs/2010.13993) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/correct_and_smooth.py)]
* **[Gini](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.functional.gini)** and **[BRO](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.functional.bro)** regularization from Henderson *et al.*: [Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity](https://arxiv.org/abs/2105.04854) (ICML 2021)
</details>

**Scalable GNNs:**
PyG supports the implementation of Graph Neural Networks that can scale to large-scale graphs.
Such application is challenging since the entire graph, its associated features and the GNN parameters cannot fit into GPU memory.
Many state-of-the-art scalability approaches tackle this challenge by sampling neighborhoods for mini-batch training, graph clustering and partitioning, or by using simplified GNN models.
These approaches have been implemented in PyG, and can benefit from the above GNN layers, operators and models.

* **[NeighborSampler](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.loader.NeighborSampler)** and **[NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.loader.NeighborLoader)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py), [**Example3**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_gat.py), [**Example4**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py)]
* **[ClusterGCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.loader.ClusterLoader)** from Chiang *et al.*: [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953) (KDD 2019) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_ppi.py)]
* **[GraphSAINT](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.loader.GraphSAINTSampler)** from Zeng *et al.*: [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931) (ICLR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_saint.py)]

<details>
<summary><b>Expand to see all implemented scalable GNNs...</b></summary>

* **[ShaDow](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.loader.ShaDowKHopSampler)** from Zeng *et al.*: [Deep Graph Neural Networks with Shallow Subgraph Samplers](https://arxiv.org/abs/2012.01380) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/shadow.py)]
* **[SIGN](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.SIGN)** from Rossi *et al.*: [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sign.py)]
* **[HGTLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.loader.HGTLoader)** from Hu *et al.*: [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) (WWW 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py)]
</details>

## Installation

### Anaconda

**Update:** You can now install PyG via [Anaconda](https://anaconda.org/pyg/pyg) for all major OS/PyTorch/CUDA combinations 🤗
Given that you have [PyTorch >= 1.8.0 installed](https://pytorch.org/get-started/locally/), simply run

```
conda install pyg -c pyg -c conda-forge
```

### Pip Wheels

We alternatively provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://data.pyg.org/whl).

#### PyTorch 1.10.0

To install the binaries for PyTorch 1.10.0, simply run

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu113` depending on your PyTorch installation (`torch.version.cuda`).

|             | `cpu` | `cu102` | `cu113` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

For additional but optional functionality, run

```
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
```

#### PyTorch 1.9.0/1.9.1

To install the binaries for PyTorch 1.9.0 and 1.9.1, simply run

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
pip install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu111` depending on your PyTorch installation (`torch.version.cuda`).

|             | `cpu` | `cu102` | `cu111` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

For additional but optional functionality, run

```
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
```

**Note:** Binaries of older versions are also provided for PyTorch 1.4.0, PyTorch 1.5.0, PyTorch 1.6.0, PyTorch 1.7.0/1.7.1 and PyTorch 1.8.0/1.8.1 (following the same procedure).

### From master

In case you want to experiment with the latest PyG features which are not fully released yet, ensure that `torch-scatter` and `torch-sparse` are installed by [following the steps mentioned above](#pip-wheels), and install PyG from master via:

```
pip install git+https://github.com/pyg-team/pytorch_geometric.git
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
If you notice anything unexpected, please open an [issue](https://github.com/pyg-team/pytorch_geometric/issues) and let us know.
If you have any questions or are missing a specific feature, feel free [to discuss them with us](https://github.com/pyg-team/pytorch_geometric/discussions).
We are motivated to constantly make PyG even better.
