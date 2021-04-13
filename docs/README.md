# Building Documentation

To build the documentation:

1. [Build and install](https://github.com/rusty1s/pytorch_geometric/blob/master/CONTRIBUTING.md#developing-pytorch-geometric) PyTorch Geometric from source.
2. Install [Sphinx](https://www.sphinx-doc.org/en/master/) via `pip install sphinx sphinx_rtd_theme`.
3. Generate the documentation file via:
```
cd pytorch_geometric/docs
make html
```
The docs are located in `pytorch_geometric/docs/build/html/index.html`.
