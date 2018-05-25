[build-image]: https://travis-ci.org/rusty1s/pytorch_geometric.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_geometric
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_geometric?branch=master

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/rusty1s/pytorch_geometric/master/docs/source/_static/img/logo.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

**[Documentation](http://rusty1s.github.io/pytorch_geometric)**

*PyTorch Geometric* is a geometric deep learning extension library for [PyTorch](https://pytorch.org/).

It consists of various methods for deep learning on graphs, known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of an elegant and easy-to-use mini-batch loader, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

In detail, the following methods are currently implemented and verified for their correctness:

* **[SplineConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SplineConv)** from Fey *et al.*: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018)
* **[GCNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GCNConv)** from Kipf and Welling: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
* **[ChebConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.ChebConv)** from Defferrard *et al.*: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) (NIPS 2017)
* **[NNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.NNConv)** adapted from Gilmer *et al.*: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) (ICML 2017)
* **[GATConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GatConv)** from Veličković *et al.*: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018)
* **[AGNNProp](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.prop.AGNNProp)** from Kiran *et al.*: [Attention-based Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735)
* **[Graclus Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007)
* **[Voxel Grid Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.voxel_grid)**

[comment]: # Head over to our documentation to find more about installation, data handling, creation of datasets and a full list of implemented methods, transformations, and datasets.

For a quick start, check out our provided [examples](https://github.com/rusty1s/pytorch_geometric/tree/master/examples) in the `examples/` directory.

We are currently in our first alpha release and work on completing [documentation](http://rusty1s.github.io/pytorch_geometric).
If you notice anything unexpected, please open a [issue](https://github.com/rusty1s/pytorch_geometric/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/rusty1s/pytorch_geometric/issues).
We are constantly encouraged to make PyTorch Geometric even better.

## Installation

If cuda is available, check that `nvcc` is accessible from your terminal, e.g. by typing `nvcc --version`.
If not, add cuda (`/usr/local/cuda/bin`) to your `$PATH`.
Then run:

```
pip install -r requirements.txt
python setup.py install
```

## Running examples

```
cd examples
python cora.py
```

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{Fey/etal/2018,
  title={{SplineCNN}: Fast Geometric Deep Learning with Continuous {B}-Spline Kernels},
  author={Fey, Matthias and Lenssen, Jan Eric and Weichert, Frank and M{\"u}ller, Heinrich},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
  year={2018},
}
```

## Running tests

```
python setup.py test
```
