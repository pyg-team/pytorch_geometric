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

PyTorch Geometric is a Geometric Deep Learning extension library for PyTorch.
In addition, PyTorch Geometric contains the spline-based convolutional layer from our paper:

Matthias Fey, Jan Eric Lenssen, Frank Weichert, Heinrich Müller: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018)

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
