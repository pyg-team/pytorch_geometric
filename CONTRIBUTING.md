# Contributing to PyG

If you are interested in contributing to PyG, your contributions will likely fall into one of the following two categories:

1. You want to implement a new feature:
   - In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or need help on the design/implementation of your feature, post about it in an issue.
2. You want to fix a bug:
   - Feel free to send a Pull Request any time you encounter a bug. Please provide a clear and concise description of what the bug was. If you are unsure about if this is a bug at all or how to fix, post about it in an issue.

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/pyg-team/pytorch_geometric.

## Developing PyG

To develop PyG on your machine, here are some tips:

1. Uninstall all existing PyG installations:

   ```bash
   pip uninstall torch-geometric
   pip uninstall torch-geometric  # run this command twice
   ```

2. Clone a copy of PyG from source:

   ```bash
   git clone https://github.com/pyg-team/pytorch_geometric
   cd pytorch_geometric
   ```

3. If you already cloned PyG from source, update it:

   ```bash
   git pull
   ```

4. Install PyG in editable mode:

   ```bash
   pip install -e ".[dev,full]"
   ```

   This mode will symlink the Python files from the current local source tree into the Python install. Hence, if you modify a Python file, you do not need to reinstall PyG again and again.

5. Ensure that you have a working PyG installation by running the entire test suite with

   ```bash
   pytest
   ```

   In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

6. Install pre-commit hooks:

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

## Building Documentation

To build the documentation:

1. [Build and install](#developing-pyg) PyG from source.
2. Install [Sphinx](https://www.sphinx-doc.org/en/master/) via `pip install sphinx sphinx_rtd_theme`.
3. Generate the documentation via:

   ```bash
   cd docs
   make html
   ```

The documentation is now available to view by opening `docs/build/html/index.html`.
