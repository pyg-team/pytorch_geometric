[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric
[build-image]: https://travis-ci.org/rusty1s/pytorch_geometric.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_geometric
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_geometric?branch=master

<p align="center">
  <img width="40%" src="https://raw.githubusercontent.com/rusty1s/pytorch_geometric/master/docs/source/_static/img/logo.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

**[Documentation](http://rusty1s.github.io/pytorch_geometric)**

*PyTorch Geometric* is a geometric deep learning extension library for [PyTorch](https://pytorch.org/).

It consists of various methods for deep learning on graphs and other irregular structures, also known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of an easy-to-use mini-batch loader, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

--------------------------------------------------------------------------------

PyTorch Geometric makes implementing graph convolutional networks a breeze (see [here](https://rusty1s.github.io/pytorch_geometric/build/html/notes/create_gnn.html) for the accompanying tutorial).
For example, this is all it takes to implement a single layer like the [edge convolution layer](https://arxiv.org/abs/1801.07829):

```python
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, F_in, F_out):
        super(EdgeConv, self).__init__()
        self.mlp = Seq(Lin(2 * F_in, F_out), ReLU(), Lin(F_out, F_out))

    def forward(self, x, edge_index):
        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(aggr='max', edge_index=edge_index, x=x)  # shape [N, F_out]

    def message(self, x_i, x_j):
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # shape [E, 2 * F_in]
        return self.mlp(edge_features)  # shape [E, F_out]
```

In addition, PyTorch Geometric is **fast**, even compared to other deep graph neural net libraries:

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Epochs</th>
      <th>Model</th>
      <th>DGL</th>
      <th>PyTorch Geometric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Cora</td>
      <td rowspan="2" align="center">200</td>
      <td align="center">GCN</td>
      <td align="right">4.2s</td>
      <td align="right"><b>0.7s</b></td>
    </tr>
    <tr>
      <td align="center">GAT</td>
      <td align="right">33.4s</td>
      <td align="right"><b>2.2s</b></td>
    </tr>
    <tr>
      <td rowspan="2">CiteSeer</td>
      <td rowspan="2" align="center">200</td>
      <td align="center">GCN</td>
      <td align="right">3.9s</td>
      <td align="right"><b>0.8s</b></td>
    </tr>
    <tr>
      <td align="center">GAT</td>
      <td align="right">28.9s</td>
      <td align="right"><b>2.4s</b></td>
    </tr>
    <tr>
      <td rowspan="2">PubMed</td>
      <td rowspan="2" align="center">200</td>
      <td align="center">GCN</td>
      <td align="right">12.7s</td>
      <td align="right"><b>2.0s</b></td>
    </tr>
    <tr>
      <td align="center">GAT</td>
      <td align="right">87.7s</td>
      <td align="right"><b>12.3s</b></td>
    </tr>
    <tr>
      <td>MUTAG</td>
      <td align="center">50</td>
      <td align="center">R-GCN</td>
      <td align="right">3.3s</td>
      <td align="right"><b>2.4s</b></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td colspan="5"><i>training runtimes obtained on a NVIDIA GTX 1080Ti</i></td>
    </tr>
  </tfoot>
</table>

--------------------------------------------------------------------------------

In detail, the following methods are currently implemented:

* **[SplineConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SplineConv)** from Fey *et al.*: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018)
* **[GCNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GCNConv)** from Kipf and Welling: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
* **[ChebConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.ChebConv)** from Defferrard *et al.*: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) (NIPS 2016)
* **[NNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.NNConv)** adapted from Gilmer *et al.*: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) (ICML 2017)
* **[GATConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GATConv)** from Veličković *et al.*: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018)
* **[AGNNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.AGNNConv)** from Thekumparampil *et al.*: [Attention-based Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735) (CoRR 2017)
* **[SAGEConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SAGEConv)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017)
* **[GraphConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GraphConv)** from, *e.g.*, Morris *et al.*: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) (AAAI 2019)
* **[GINConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GINConv)** from Xu *et al.*: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (ICLR 2019)
* **[RGCNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.RGCNConv)** from Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (ESWC 2018)
* **[EdgeConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.EdgeConv)** from Wang *et al.*: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) (CoRR, 2018)
* **[PointConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.PointConv)** (including **[Iterative Farthest Point Sampling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.fps)** and dynamic graph generation based on **[nearest neighbor](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.knn_graph)** or **[maximum distance](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.radius_graph)**) from Qi *et al.*: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) (CVPR 2017) and [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) (NIPS 2017)
* **[XConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.XConv)** from Li *et al.*: [PointCNN: Convolution On X-Transformed Points](https://arxiv.org/abs/1801.07791) (NeurIPS 2018)
* **[GMMConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GMMConv)** from Monti *et al.*: [Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs](https://arxiv.org/abs/1612.00593) (CVPR 2017)
* A **[MetaLayer](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.meta.MetaLayer)** for building any kind of graph network similar to the [TensorFlow Graph Nets library](https://github.com/deepmind/graph_nets) from Battaglia *et al.*: [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) (CoRR 2018)
* **[GlobalAttention](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.glob.GlobalAttention)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016)
* **[Set2Set](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.glob.Set2Set)** from Vinyals *et al.*: [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391) (ICLR 2016)
* **[Sort Pool](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.glob.global_sort_pool)** from Zhang *et al.*: [An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) (AAAI 2018)
* **[Dense Differentiable Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.dense.diff_pool.dense_diff_pool)** from Ying *et al.*: [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) (NeurIPS 2018)
* **[Graclus Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007)
* **[Voxel Grid Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.voxel_grid)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)
* **[Top-K Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.TopKPooling)** from Gao and Ji: [Graph U-Net](https://openreview.net/forum?id=HJePRoAct7) (ICLR 2019 submission) and Cangea *et al.*: [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287) (NeurIPS-W 2018)
* Example of **[Deep Graph Infomax on Cora](https://github.com/rusty1s/pytorch_geometric/tree/master/examples/infomax.py)** from Veličković *et al.*: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (ICLR 2019)

--------------------------------------------------------------------------------

Head over to our [documentation](http://rusty1s.github.io/pytorch_geometric) to find out more about installation, data handling, creation of datasets and a full list of implemented methods, transforms, and datasets.
For a quick start, check out our provided [examples](https://github.com/rusty1s/pytorch_geometric/tree/master/examples) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](https://github.com/rusty1s/pytorch_geometric/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/rusty1s/pytorch_geometric/issues).
We are constantly encouraged to make PyTorch Geometric even better.

## Installation

Ensure that at least PyTorch 1.0.0 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 1.0.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```

Then run:

```
$ pip install --upgrade torch-scatter
$ pip install --upgrade torch-sparse
$ pip install --upgrade torch-cluster
$ pip install --upgrade torch-spline-conv
$ pip install torch-geometric
```

If you are running into any installation problems, please create an [issue](https://github.com/rusty1s/pytorch_geometric/issues).
Beforehand, please check that the [official extension example](https://github.com/pytorch/extension-cpp) runs on your machine.

## Running examples

```
cd examples
python cora.py
```

## Cite

Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:

```
@inproceedings{Fey/etal/2018,
  title={{SplineCNN}: Fast Geometric Deep Learning with Continuous {B}-Spline Kernels},
  author={Fey, Matthias and Lenssen, Jan Eric and Weichert, Frank and M{\"u}ller, Heinrich},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
}
```

## Running tests

```
python setup.py test
```
