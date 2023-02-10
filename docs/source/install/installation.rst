Installation
============

:pyg:`PyG` is available for Python 3.7 to Python 3.10.

.. note::
   We do not recommend installation as a root user on your system Python.
   Please setup a virtual environment, *e.g.*, via `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_, or create a `Docker image <https://www.docker.com/>`_.

Quick Start
-----------

.. raw:: html
   :file: quick-start.html

Installation via Anaconda
-------------------------

You can now install :pyg:`PyG` via `Anaconda <https://anaconda.org/pyg/pyg>`_ for all major OS/PyTorch/CUDA combinations 🤗
If you have not yet installed :pytorch:`PyTorch`, install it via :obj:`conda` as described in its `official documentation <https://pytorch.org/get-started/locally/>`_.
Given that you have :pytorch:`PyTorch` installed (:obj:`>=1.8.0`), simply run

.. code-block:: none

   conda install pyg -c pyg

.. warning::
   Conda packages are currently not available for M1/M2/M3 macs.
   Please install the extension packages `from source <installation.html#installation-from-source>`__.

Installation via Pip Wheels
---------------------------

We have outsourced a lot of functionality of :pyg:`PyG` to other packages, which needs to be installed in advance.
These packages come with their own CPU and GPU kernel implementations based on the `PyTorch C++/CUDA extension interface <https://github.com/pytorch/extension-cpp/>`_.
We provide pip wheels for these packages for all major OS/PyTorch/CUDA combinations, see `here <https://data.pyg.org/whl>`__:

.. warning::
   Wheels are currently not available for M1/M2/M3 macs.
   Please install the extension packages `from source <installation.html#installation-from-source>`__.

#. Ensure that at least :pytorch:`PyTorch` 1.12.0 is installed:

   .. code-block:: none

      python -c "import torch; print(torch.__version__)"
      >>> 1.13.0

#. Find the CUDA version :pytorch:`PyTorch` was installed with:

   .. code-block:: none

      python -c "import torch; print(torch.version.cuda)"
      >>> 11.6

#. Install the relevant packages:

   .. code-block:: none

      pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
      pip install torch-geometric

   where :obj:`${TORCH}` and :obj:`${CUDA}` should be replaced by the specific :pytorch:`PyTorch` and CUDA versions, respectively:

   * :pytorch:`PyTorch` 1.13: :obj:`${TORCH}=1.13.0` and :obj:`${CUDA}=cpu|cu116|cu117`
   * :pytorch:`PyTorch` 1.12: :obj:`${TORCH}=1.12.0` and :obj:`${CUDA}=cpu|cu102|cu113|cu116`

   For example, for :pytorch:`PyTorch` 1.13.* and CUDA 11.6, type:

   .. code-block:: none

      pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
      pip install torch-geometric

   For :pytorch:`PyTorch` 1.12.* and CUDA 11.3, type:

   .. code-block:: none

     pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
     pip install torch-geometric

#. Install additional packages *(optional)*:

   To add additional functionality to :pyg:`PyG`, such as k-NN and radius graph generation or :class:`~torch_geometric.nn.conv.SplineConv` support, run

   .. code-block:: none

      pip install torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

   following the same procedure as mentioned above.

**Note:** Binaries of older versions are also provided for :pytorch:`PyTorch` 1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.10.0, 1.10.1, 1.10.2, and 1.11.0 (following the same procedure).
**For older versions, you need to explicitly specify the latest supported version number** or install via :obj:`pip install --no-index` in order to prevent a manual installation from source.
You can look up the latest supported version number `here <https://data.pyg.org/whl>`__.

Installation from Source
------------------------

In case a specific version is not supported by `our wheels <https://data.pyg.org/whl>`_, you can alternatively install :pyg:`PyG` from source:

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
         >>> 11.3

   #. Ensure that :pytorch:`PyTorch` and system CUDA versions match:

      .. code-block:: none

         python -c "import torch; print(torch.version.cuda)"
         >>> 11.3

         nvcc --version
         >>> 11.3

#. Install the relevant packages:

   .. code-block:: none

      pip install git+https://github.com/pyg-team/pyg-lib.git
      pip install torch-scatter
      pip install torch-sparse
      pip install torch-geometric

#. Install additional packages *(optional)*:

   .. code-block:: none

      pip install torch-cluster
      pip install torch-spline-conv

In rare cases, CUDA or Python path problems can prevent a successful installation.
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
