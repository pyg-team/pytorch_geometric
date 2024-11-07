Installation
============

:pyg:`PyG` is available for :python:`Python 3.9` to :python:`Python 3.12`.

.. note::
   We do not recommend installation as a root user on your system :python:`Python`.
   Please setup a virtual environment, *e.g.*, via :conda:`null` `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_, or create a `Docker image <https://www.docker.com/>`_.

Quick Start
-----------

.. raw:: html
   :file: quick-start.html

Installation via Anaconda
-------------------------

You can now install :pyg:`PyG` via :conda:`null` `Anaconda <https://anaconda.org/pyg/pyg>`_ for all major OS, :pytorch:`PyTorch` and CUDA combinations 🤗.
If you have not yet installed :pytorch:`PyTorch`, install it via :conda:`null` :obj:`conda install` as described in its `official documentation <https://pytorch.org/get-started/locally/>`_.
Given that you have :pytorch:`PyTorch` installed (:obj:`>=1.11.0`), simply run

.. code-block:: none

   conda install pyg -c pyg

.. warning::
   Conda packages are currently not available for Windows and M1/M2/M3 macs.

If :conda:`null` :obj:`conda` does not pick up the correct CUDA version of :pyg:`PyG`, you can enforce it as follows:

.. code-block:: none

   conda install pyg=*=*cu* -c pyg

Installation via PyPi
---------------------

From :pyg:`null` **PyG 2.3** onwards, you can install and use :pyg:`PyG` **without any external library** required except for :pytorch:`PyTorch`.
For this, simply run:

.. code-block:: none

   pip install torch_geometric

Additional Libraries
--------------------

If you want to utilize the full set of features from :pyg:`PyG`, there exists several additional libraries you may want to install:

* `pyg-lib <https://github.com/pyg-team/pyg-lib>`__: Heterogeneous GNN operators and graph sampling routines
* `torch-scatter <https://github.com/rusty1s/pytorch_scatter>`__: Accelerated and efficient sparse reductions
* `torch-sparse <https://github.com/rusty1s/pytorch_sparse>`__: :class:`SparseTensor` support, see `here <https://pytorch-geometric.readthedocs.io/en/latest/advanced/sparse_tensor.html>`__
* `torch-cluster <https://github.com/rusty1s/pytorch_cluster>`__: Graph clustering routines
* `torch-spline-conv <https://github.com/rusty1s/pytorch_spline_conv>`__: :class:`~torch_geometric.nn.conv.SplineConv` support

These packages come with their own CPU and GPU kernel implementations based on the :pytorch:`null` `PyTorch C++/CUDA/hip(ROCm) extension interface <https://github.com/pytorch/extension-cpp/>`_.
For a basic usage of :pyg:`PyG`, these dependencies are **fully optional**.
We recommend to start with a minimal installation, and install additional dependencies once you start to actually need them.

Installation from Wheels
~~~~~~~~~~~~~~~~~~~~~~~~

For ease of installation of these extensions, we provide :obj:`pip` wheels for these packages for all major OS, :pytorch:`PyTorch` and CUDA combinations, see `here <https://data.pyg.org/whl>`__:

.. warning::
   Wheels are currently not available for M1/M2/M3 macs.
   Please install the extension packages `from source <installation.html#installation-from-source>`__.

#. Ensure that at least :pytorch:`PyTorch` 1.13.0 is installed:

   .. code-block:: none

      python -c "import torch; print(torch.__version__)"
      >>> 2.4.0

#. Find the CUDA version :pytorch:`PyTorch` was installed with:

   .. code-block:: none

      python -c "import torch; print(torch.version.cuda)"
      >>> 12.4

#. Install the relevant packages:

   .. code-block:: none

      pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

   where :obj:`${TORCH}` and :obj:`${CUDA}` should be replaced by the specific :pytorch:`PyTorch` and CUDA versions, respectively:

   * :pytorch:`PyTorch` 2.4: :obj:`${TORCH}=2.4.0` and :obj:`${CUDA}=cpu|cu118|cu121|cu124`
   * :pytorch:`PyTorch` 2.3: :obj:`${TORCH}=2.3.0` and :obj:`${CUDA}=cpu|cu118|cu121`
   * :pytorch:`PyTorch` 2.2: :obj:`${TORCH}=2.2.0` and :obj:`${CUDA}=cpu|cu118|cu121`
   * :pytorch:`PyTorch` 2.1: :obj:`${TORCH}=2.1.0` and :obj:`${CUDA}=cpu|cu118|cu121`
   * :pytorch:`PyTorch` 2.0: :obj:`${TORCH}=2.0.0` and :obj:`${CUDA}=cpu|cu117|cu118`
   * :pytorch:`PyTorch` 1.13: :obj:`${TORCH}=1.13.0` and :obj:`${CUDA}=cpu|cu116|cu117`

   For example, for :pytorch:`PyTorch` 2.4.* and CUDA 12.4, type:

   .. code-block:: none

      pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

   For example, for :pytorch:`PyTorch` 2.3.* and CUDA 11.8, type:

   .. code-block:: none

      pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

**Note:** Binaries of older versions are also provided for :pytorch:`PyTorch` 1.4.0, 1.5.0, 1.6.0, 1.7.0/1.7.1, 1.8.0/1.8.1, 1.9.0, 1.10.0/1.10.1/1.10.2, 1.11.0, 1.12.0/1.12.1, 1.13.0/1.13.1, 2.0.0/2.0.1, 2.1.0/2.1.1/2.1.2, and 2.2.0/2.2.1/2.2.2 (following the same procedure).
**For older versions, you need to explicitly specify the latest supported version number** or install via :obj:`pip install --no-index` in order to prevent a manual installation from source.
You can look up the latest supported version number `here <https://data.pyg.org/whl>`__.

**ROCm:** The external `pyg-rocm-build repository <https://github.com/Looong01/pyg-rocm-build>`__ provides wheels and detailed instructions on how to install :pyg:`PyG` for ROCm.
If you have any questions about it, please open an issue `here <https://github.com/Looong01/pyg-rocm-build/issues>`__.

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

In case a specific version is not supported by `our wheels <https://data.pyg.org/whl>`_, you can alternatively install them from source:

#. Ensure that your CUDA is setup correctly (optional):

   #. Check if :pytorch:`PyTorch` is installed with CUDA support:

      .. code-block:: none

         python -c "import torch; print(torch.cuda.is_available())"
         >>> True

   #. Add CUDA to :obj:`$PATH` and :obj:`$CPATH` (note that your actual CUDA path may vary from :obj:`/usr/local/cuda`):

      .. code-block:: none

         export PATH=/usr/local/cuda/bin:$PATH
         echo $PATH
         >>> /usr/local/cuda/bin:...

         export CPATH=/usr/local/cuda/include:$CPATH
         echo $CPATH
         >>> /usr/local/cuda/include:...

   #. Add CUDA to :obj:`$LD_LIBRARY_PATH` on Linux and to :obj:`$DYLD_LIBRARY_PATH` on macOS (note that your actual CUDA path may vary from :obj:`/usr/local/cuda`):

      .. code-block:: none

         export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
         echo $LD_LIBRARY_PATH
         >>> /usr/local/cuda/lib64:...

         export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
         echo $DYLD_LIBRARY_PATH
         >>> /usr/local/cuda/lib:...

   #. Verify that :obj:`nvcc` is accessible from terminal:

      .. code-block:: none

         nvcc --version
         >>> 11.8

   #. Ensure that :pytorch:`PyTorch` and system CUDA versions match:

      .. code-block:: none

         python -c "import torch; print(torch.version.cuda)"
         >>> 11.8

         nvcc --version
         >>> 11.8

#. Install the relevant packages:

   .. code-block:: none

      pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
      pip install --verbose torch_scatter
      pip install --verbose torch_sparse
      pip install --verbose torch_cluster
      pip install --verbose torch_spline_conv

In rare cases, CUDA or :python:`Python` path problems can prevent a successful installation.
:obj:`pip` may even signal a successful installation, but execution simply crashes with :obj:`Segmentation fault (core dumped)`.
We collected common installation errors in the `Frequently Asked Questions <installation.html#frequently-asked-questions>`__ subsection.
In case the FAQ does not help you in solving your problem, please create an `issue <https://github.com/pyg-team/pytorch_geometric/issues>`_.
Before, please verify that your CUDA is set up correctly by following the official `installation guide <https://docs.nvidia.com/cuda>`_.

Frequently Asked Questions
--------------------------

#. :obj:`undefined symbol: **make_function_schema**`: This issue signals (1) a **version conflict** between your installed :pytorch:`PyTorch` version and the :obj:`${TORCH}` version specified to install the extension packages, or (2) a version conflict between the installed CUDA version of :pytorch:`PyTorch` and the :obj:`${CUDA}` version specified to install the extension packages.
   Please verify that your :pytorch:`PyTorch` version and its CUDA version **match** with your installation command:

   .. code-block:: none

      python -c "import torch; print(torch.__version__)"
      python -c "import torch; print(torch.version.cuda)"
      nvcc --version

   For re-installation, ensure that you do not run into any caching issues by using the :obj:`pip --force-reinstall --no-cache-dir` flags.
   In addition, the :obj:`pip --verbose` option may help to track down any issues during installation.
   If you still do not find any success in installation, please try to install the extension packages `from source <installation.html#installation-from-source>`__.
