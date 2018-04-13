<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/rusty1s/pytorch_geometric/master/docs/source/_static/img/logo.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

PyTorch Geometric is a Geometric Deep Learning extension library for PyTorch.
In addition, PyTorch Geometric contains the spline-based convolutional layer from our paper [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920).

**[Documentation](http://rusty1s.github.io/pytorch_geometric)**

![Development Status][development-status-image]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

[development-status-image]: https://img.shields.io/badge/status-in%20development-blue.svg
[build-image]: https://travis-ci.org/rusty1s/pytorch_geometric.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_geometric
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_geometric?branch=master

## Requirements

Install `pytorch` and `torchvision` manually ([Installation](http://pytorch.org/)).
If cuda is available it is recommended to install `cupy` ([Installation](https://github.com/cupy/cupy#installation)).

Run the following command to install the additional dependencies:

```shell
pip install -r requirements.txt
```

## Running tests

Install the test requirements and run the test suite:

```shell
pip install -r requirements_test.txt
nosetests
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
