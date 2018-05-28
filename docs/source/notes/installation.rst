Installation
============

We have outsourced a lot of functionality of `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ to other packages, which needs to be installed in advance.
These packages come with their own CPU and GPU kernel implementations based on `C FFI <https://github.com/pytorch/extension-ffi/>`_ and the newly introduced `C++/CUDA extensions <https://github.com/pytorch/extension-cpp/>`_ in PyTorch 0.4.0.

.. note::
    We do not recommend installation as root user on your system python.
    Please setup an `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment or `Docker image <https://www.docker.com/>`_.

Please follow the steps below for a successful installation:

#. Ensure that PyTorch 0.4.0 is installed:

    .. code-block:: none

        $ python -c "import torch; print(torch.__version__)"

#. Ensure CUDA is setup correctly (optional):

    #. Check if PyTorch is installed with CUDA support:

        .. code-block:: none

            $ python -c "import torch; print(torch.cuda.is_available())"

    #. Add CUDA to ``$PATH`` and ``$CPATH`` (note that your actual CUDA path may vary from ``/usr/local/cuda``):

        .. code-block:: none

            $ PATH=/usr/local/cuda/bin:$PATH
            $ echo $PATH

            $ CPATH=/usr/local/cuda/install:$CPATH
            $ echo $CPATH

    #. Verify that ``nvcc`` is accessible from terminal:

        .. code-block:: none

            $ nvcc --version

#. Install all needed packages:

    .. code-block:: none

        $ pip install cffi
        $ pip install --upgrade torch-scatter
        $ pip install --upgrade torch-unique
        $ pip install --upgrade torch-cluster
        $ pip install --upgrade torch-spline-conv
        $ pip install torch-geometric

In rare cases, CUDA or python path issues can prevent a succesful installation.
Unfortunately, the error messages of ``pip`` are not very meaningful.
You should therefore clone the respective package and check where the error occurs, e.g.:

.. code-block:: none

    $ git clone https://github.com/rusty1s/pytorch_scatter
    $ cd pytorch_scatter
    $ ./build.sh  # Check for CUDA compilation error.
    $ python setup.py install  # Check for link error.
    $ python setup.py test  # Verify.

C++/CUDA extensions on macOS
----------------------------

In order to compile C++/CUDA extensions on macOS, you need to replace the call

.. code-block:: python

    def spawn(self, cmd):
        spawn(cmd, dry_run=self.dry_run)

with

.. code-block:: python

    def spawn(self, cmd):
        subprocess.call(cmd)

in ``distutils/ccompiler.py``.
Do not forget to ``import subprocess`` at the top of the file.
