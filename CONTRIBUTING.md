# Contributing to PyG

If you are interested in contributing to PyG, your contributions will likely fall into one of the following two categories:

1. You want to implement a new feature:
   - In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or need help on the design/implementation of your feature, post about it in an issue.
2. You want to fix a bug:
   - Feel free to send a Pull Request any time you encounter a bug. Please provide a clear and concise description of what the bug was. If you are unsure about if this is a bug at all or how to fix, post about it in an issue.

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/pyg-team/pytorch_geometric.

## Developing PyG

To develop PyG on your machine, here are some tips:

1. Ensure that you are running on one of the two latest PyTorch releases (*e.g.*, `1.12.0`):

   ```python
   import torch
   print(torch.__version__)
   ```

2. Follow the [installation instructions](https://github.com/pyg-team/pytorch_geometric#installation) to install `torch-scatter`, `torch-sparse`, `torch-cluster` and `torch-spline-conv` (if you haven't already):

   ```bash
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   ```

   where `${TORCH}` should be replaced by your PyTorch version (*e.g.*, `1.12.0`), and `${CUDA}` should be replaced by your CUDA version (*e.g.*, `cpu` or `cu116`).

3. Uninstall all existing PyG installations:

   ```bash
   pip uninstall torch-geometric
   pip uninstall torch-geometric  # run this command twice
   ```

4. Clone a copy of PyG from source:

   ```bash
   git clone https://github.com/pyg-team/pytorch_geometric
   cd pytorch_geometric
   ```

5. If you already cloned PyG from source, update it:

   ```bash
   git pull
   ```

6. Install PyG in editable mode:

   ```bash
   pip install -e ".[dev,full]"
   ```

   This mode will symlink the Python files from the current local source tree into the Python install. Hence, if you modify a Python file, you do not need to reinstall PyG again and again.

7. Ensure that you have a working PyG installation by running the entire test suite with

   ```bash
   pytest
   ```

   In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

8. Install pre-commit hooks:

   ```bash
    pre-commit install
   ```

## Unit Testing

The PyG testing suite is located under `test/`.
Run the entire test suite with

```bash
pytest
```

or test individual files via, _e.g._, `pytest test/utils/test_convert.py`.

## Continuous Integration

PyG uses [GitHub Actions](https://github.com/pyg-team/pytorch_geometric/actions) in combination with [CodeCov](https://codecov.io/github/pyg-team/pytorch_geometric?branch=master) for continuous integration.

Everytime you send a Pull Request, your commit will be built and checked against the PyG guidelines:

1. Ensure that your code is formatted correctly by testing against the styleguide of [`flake8`](https://github.com/PyCQA/flake8)://github.com/PyCQA/pycodestyle):

   ```bash
   flake8 .
   ```

   If you do not want to format your code manually, we recommend to use [`yapf`](https://github.com/google/yapf).

2. Ensure that the entire test suite passes and that code coverage roughly stays the same.
   Please feel encouraged to provide a test with your submitted code.
   To test, either run

   ```bash
   pytest --cov
   ```

   or

   ```bash
   FULL_TEST=1 pytest --cov
   ```

   (which runs a set of additional but time-consuming tests) dependening on your needs.

3. Add your feature/bugfix to the [`CHANGELOG.md`](https://github.com/pyg-team/pytorch_geometric/blob/master/CHANGELOG.md?plain=1).
   If multiple PRs move towards integrating a single feature, it is advised to group them together into one bullet point.

## Building Documentation

To build the documentation:

1. [Build and install](#developing-pyg) PyG from source.
2. Install [Sphinx](https://www.sphinx-doc.org/en/master/) theme via
   ```bash
   pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
   ```
3. Generate the documentation via:
   ```bash
   cd docs
   make html
   ```

The documentation is now available to view by opening `docs/build/html/index.html`.
