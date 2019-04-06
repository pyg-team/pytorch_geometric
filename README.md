[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric
[build-image]: https://travis-ci.org/rusty1s/pytorch_geometric.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_geometric
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_geometric?branch=master

<p align="center">
  <img width="40%" src="https://raw.githubusercontent.com/rusty1s/pytorch_geometric/master/docs/source/_static/img/pyg_logo_text.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

**[Documentation](http://rusty1s.github.io/pytorch_geometric)** | **[Paper](https://arxiv.org/abs/1903.02428)**

*PyTorch Geometric* (PyG) is a geometric deep learning extension library for [PyTorch](https://pytorch.org/).

It consists of various methods for deep learning on graphs and other irregular structures, also known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of an easy-to-use mini-batch loader, multi gpu-support, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

--------------------------------------------------------------------------------

PyTorch Geometric makes implementing Graph Neural Networks a breeze (see [here](https://rusty1s.github.io/pytorch_geometric/build/html/notes/create_gnn.html) for the accompanying tutorial).
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

In addition, PyTorch Geometric is **fast**, even compared to other deep graph neural net libraries:

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Epochs</th>
      <th>Model</th>
      <th>DGL-DB</th>
      <th>DGL-SPMV</th>
      <th>PyG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Cora</td>
      <td rowspan="2" align="center">200</td>
      <td align="center">GCN</td>
      <td align="right">4.19s</td>
      <td align="right">0.32s</td>
      <td align="right"><b>0.25s</b></td>
    </tr>
    <tr>
      <td align="center">GAT</td>
      <td align="right">6.31s</td>
      <td align="right">5.36s</td>
      <td align="right"><b>0.80s</b></td>
    </tr>
    <tr>
      <td rowspan="2">CiteSeer</td>
      <td rowspan="2" align="center">200</td>
      <td align="center">GCN</td>
      <td align="right">3.78s</td>
      <td align="right">0.34s</td>
      <td align="right"><b>0.30s</b></td>
    </tr>
    <tr>
      <td align="center">GAT</td>
      <td align="right">5.61s</td>
      <td align="right">4.91s</td>
      <td align="right"><b>0.88s</b></td>
    </tr>
    <tr>
      <td rowspan="2">PubMed</td>
      <td rowspan="2" align="center">200</td>
      <td align="center">GCN</td>
      <td align="right">12.91s</td>
      <td align="right">0.36s</td>
      <td align="right"><b>0.32s</b></td>
    </tr>
    <tr>
      <td align="center">GAT</td>
      <td align="right">18.69s</td>
      <td align="right">13.76s</td>
      <td align="right"><b>2.42s</b></td>
    </tr>
    <tr>
      <td>MUTAG</td>
      <td align="center">200</td>
      <td align="center">RGCN</td>
      <td align="right">18.81s</td>
      <td align="right">2.40s</td>
      <td align="right"><b>2.14s</b></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td colspan="6"><b>DGL-DB:</b> DGL 0.2 with <i>Degree Bucketing</i>.</td>
    </tr>
    <tr>
      <td colspan="6"><b>DGL-SPMV:</b> DGL 0.2 with <i>SPMV optimization</i> inspired by PyG.</td>
    </tr>
    <tr>
      <td colspan="6">Training runtimes obtained on a NVIDIA GTX 1080Ti.</td>
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
* **[SAGEConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SAGEConv)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017)
* **[GraphConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GraphConv)** from, *e.g.*, Morris *et al.*: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) (AAAI 2019)
* **[GatedGraphConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GatedGraphConv)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016)
* **[GINConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GINConv)** from Xu *et al.*: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (ICLR 2019)
* **[ARMAConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.ARMAConv)** from Bianchi *et al.*: [Graph Neural Networks with Convolutional ARMA Filters](https://arxiv.org/abs/1901.01343) (CoRR 2019)
* **[SGConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SGConv)** from Wu *et al.*: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) (CoRR 2019)
* **[APPNP](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.APPNP)** from Klicpera *et al.*: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997) (ICLR 2019)
* **[AGNNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.AGNNConv)** from Thekumparampil *et al.*: [Attention-based Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735) (CoRR 2017)
* **[RGCNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.RGCNConv)** from Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (ESWC 2018)
* **[EdgeConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.EdgeConv)** from Wang *et al.*: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) (CoRR, 2018)
* **[PointConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.PointConv)** (including **[Iterative Farthest Point Sampling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.fps)**, dynamic graph generation based on **[nearest neighbor](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.knn_graph)** or **[maximum distance](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.radius_graph)**, and **[k-NN interpolation](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.unpool.knn_interpolate)**) from Qi *et al.*: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) (CVPR 2017) and [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) (NIPS 2017)
* **[XConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.XConv)** from Li *et al.*: [PointCNN: Convolution On X-Transformed Points](https://arxiv.org/abs/1801.07791) [(official implementation)](https://github.com/yangyanli/PointCNN) (NeurIPS 2018)
* **[GMMConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GMMConv)** from Monti *et al.*: [Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs](https://arxiv.org/abs/1611.08402) (CVPR 2017)
* A **[MetaLayer](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.meta.MetaLayer)** for building any kind of graph network similar to the [TensorFlow Graph Nets library](https://github.com/deepmind/graph_nets) from Battaglia *et al.*: [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) (CoRR 2018)
* **[GlobalAttention](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.glob.GlobalAttention)** from Li *et al.*: [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (ICLR 2016)
* **[Set2Set](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.glob.Set2Set)** from Vinyals *et al.*: [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391) (ICLR 2016)
* **[Sort Pool](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.glob.global_sort_pool)** from Zhang *et al.*: [An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) (AAAI 2018)
* **[Dense Differentiable Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.dense.diff_pool.dense_diff_pool)** from Ying *et al.*: [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) (NeurIPS 2018)
* **[Graclus Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007)
* **[Voxel Grid Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.voxel_grid)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)
* **[Top-K Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.TopKPooling)** from Gao and Ji: [Graph U-Net](https://openreview.net/forum?id=HJePRoAct7) (ICLR 2019 submission) and Cangea *et al.*: [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287) (NeurIPS-W 2018)
* **[Local Degree Profile](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.transforms.LocalDegreeProfile)** from Cai and Wang: [A Simple yet Effective Baseline for Non-attribute Graph Classification](https://arxiv.org/abs/1811.03508) (CoRR 2018)
* All variants of **[Graph Auto-Encoders](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.models.GAE)** from Kipf and Welling: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) (NIPS-W 2016) and Pan *et al.*: [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/abs/1802.04407) (IJCAI 2018)
* Example of **[Deep Graph Infomax on Cora](https://github.com/rusty1s/pytorch_geometric/tree/master/examples/infomax.py)** from Veličković *et al.*: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (ICLR 2019)

--------------------------------------------------------------------------------

Head over to our [documentation](http://rusty1s.github.io/pytorch_geometric) to find out more about installation, data handling, creation of datasets and a full list of implemented methods, transforms, and datasets.
For a quick start, check out our [examples](https://github.com/rusty1s/pytorch_geometric/tree/master/examples) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](https://github.com/rusty1s/pytorch_geometric/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/rusty1s/pytorch_geometric/issues).
We are constantly encouraged to make PyTorch Geometric even better.

## Installation

Ensure that at least PyTorch 1.0.0 is installed and verify that `cuda/bin`, `cuda/include` and `cuda/lib64` are in your `$PATH`, `$CPATH` and `$LD_LIBRARY_PATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 1.0.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```
and
```
$ echo $LD_LIBRARY_PATH
>>> /usr/local/cuda/lib64
```
on Linux or
```
$ echo $DYLD_LIBRARY_PATH
>>> /usr/local/cuda/lib
```
on macOS.
Then run:

```sh
$ pip install --verbose --no-cache-dir torch-scatter
$ pip install --verbose --no-cache-dir torch-sparse
$ pip install --verbose --no-cache-dir torch-cluster
$ pip install --verbose --no-cache-dir torch-spline-conv (optional)
$ pip install torch-geometric
```

See the [Frequently Asked Questions](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html#frequently-asked-questions) and verify that your [CUDA is set up correctly](https://docs.nvidia.com/cuda/index.html) if you encounter any installation errors.

## Running examples

```
cd examples
python cora.py
```

## Cite

Please cite [our paper](https://arxiv.org/abs/1903.02428) (and the respective papers of the methods used) if you use this code in your own work:

```
@article{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  journal={CoRR},
  volume={abs/1903.02428},
  year={2019},
}
```

## Running tests

```
python setup.py test
```
