<p align="center">
  <img width="80%" src="https://user-images.githubusercontent.com/6945922/32413706-b83b5530-c217-11e7-83ef-43da79b8b5aa.png" />
</p>

--------------------------------------------------------------------------------

PyTorch Geometric is a Geometric Deep Learning extension library for PyTorch

![Development Status][development-status-image]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Requirements Status][requirements-image]][requirements-url]
[![Code Climate][code-climate-image]][code-climate-url]

[development-status-image]: https://img.shields.io/badge/status-in%20development-blue.svg
[build-image]: https://travis-ci.org/rusty1s/pytorch_geometric.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_geometric
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_geometric/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_geometric?branch=master
[requirements-image]: https://requires.io/github/rusty1s/pytorch_geometric/requirements.svg?branch=master
[requirements-url]: https://requires.io/github/rusty1s/pytorch_geometric/requirements/?branch=master
[code-climate-image]: https://api.codeclimate.com/v1/badges/ea4a011808e1bbcd5ea7/maintainability
[code-climate-url]: https://codeclimate.com/github/rusty1s/pytorch_geometric

## Requirements

Install `pytorch` and `torchvision` manually ([Installation](http://pytorch.org/)).
If cuda is available it is recommended to install `cupy` ([Installation](https://github.com/cupy/cupy#installation)).

To install the additional python packages needed, run:

```shell
pip install -r requirements.txt
```

## Running tests

Install the test requirements and run the test suite:

```shell
pip install -r requirements_test.txt
nosetests
```
