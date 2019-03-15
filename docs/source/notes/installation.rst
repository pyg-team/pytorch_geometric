Installation
============

We have outsourced a lot of functionality of `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ to other packages, which needs to be installed in advance.
These packages come with their own CPU and GPU kernel implementations based on the newly introduced `C++/CUDA extensions <https://github.com/pytorch/extension-cpp/>`_ in PyTorch 0.4.0.

.. note::
    We do not recommend installation as root user on your system python.
    Please setup an `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment or create a `Docker image <https://www.docker.com/>`_.

Please follow the steps below for a successful installation:

#. Ensure that at least PyTorch 1.0.0 is installed:

    .. code-block:: none

        $ python -c "import torch; print(torch.__version__)"
        >>> 1.0.0

#. Ensure CUDA is setup correctly (optional):

    #. Check if PyTorch is installed with CUDA support:

        .. code-block:: none

            $ python -c "import torch; print(torch.cuda.is_available())"
            >>> True

    #. Add CUDA to ``$PATH`` and ``$CPATH`` (note that your actual CUDA path may vary from ``/usr/local/cuda``):

        .. code-block:: none

            $ PATH=/usr/local/cuda/bin:$PATH
            $ echo $PATH
            >>> /usr/local/cuda/bin:...

            $ CPATH=/usr/local/cuda/include:$CPATH
            $ echo $CPATH
            >>> /usr/local/cuda/include:...

    #. Add CUDA to ``$LD_LIBRARY_PATH`` on Linux and to ``$DYLD_LIBRARY_PATH`` on macOS (note that your actual CUDA path may vary from ``/usr/local/cuda``):

        .. code-block:: none

            $ LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
            $ echo $LD_LIBRARY_PATH
            >>> /usr/local/cuda/lib64:...

            $ DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
            $ echo $DYLD_LIBRARY_PATH
            >>> /usr/local/cuda/lib:...

    #. Verify that ``nvcc`` is accessible from terminal:

        .. code-block:: none

            $ nvcc --version
            >>> 10.0

    #. Ensure that PyTorch and system CUDA versions match:

        .. code-block:: none

            $ python -c "import torch; print(torch.version.cuda)"
            >>> 10.0

            $ nvcc --version
            >>> 10.0

#. Install all needed packages:

    .. code-block:: none

        $ pip install --upgrade torch-scatter
        $ pip install --upgrade torch-sparse
        $ pip install --upgrade torch-cluster
        $ pip install --upgrade torch-spline-conv (optional)
        $ pip install torch-geometric

In rare cases, CUDA or python path issues can prevent a successful installation.
You can verify that your CUDA is set up correctly by following the offical `installation guides <https://docs.nvidia.com/cuda/index.html>`_.
If an error occurs during installation, ``pip`` may not issue a reasonable error message or even pretend that everything went well (resulting in runtime errors).
You should then clone the respective package and check the error message by installing from source, *e.g.*:

.. code-block:: none

    $ git clone https://github.com/rusty1s/pytorch_scatter
    $ cd pytorch_scatter
    $ python setup.py install  # Check for CUDA compilation or link error.
    $ python setup.py test  # Verify installation by running test suite.

We collected a lot of common installation errors in the `Frequently Asked Questions <https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html#frequently-asked-questions>`_ subsection.
Please note that if you make any changes, you need to force a recompilation of the CUDA kernels by removing the ``build/`` directory in the root folder of the respective package.
In case the FAQ does not help you in solving your problem, please create an `issue <https://github.com/rusty1s/pytorch_geometric/issues>`_, please create an `issue <https://github.com/rusty1s/pytorch_geometric/issues>`_.
You can additionaly check that the `official extension example <https://github.com/pytorch/extension-cpp>`_ runs on your machine.

C++/CUDA Extensions on macOS
----------------------------

In order to compile CUDA extensions on macOS, you need to replace the call

.. code-block:: python

    def spawn(self, cmd):
        spawn(cmd, dry_run=self.dry_run)

with

.. code-block:: python

    import subprocess

    def spawn(self, cmd):
        subprocess.call(cmd)

in ``lib/python{xxx}/distutils/ccompiler.py``.

Frequently Asked Questions
--------------------------

#. ``ImportError: ***: cannot open shared object file: No such file or directory``: Add CUDA to your ``$LD_LIBRARY_PATH`` (see `Issue#43 <https://github.com/rusty1s/pytorch_geometric/issues/43>`_).

#. ``undefined symbol:``, *e.g.* ``_ZN2at6detail20DynamicCUDAInterface10set_deviceE``: Clear the pip cache and reinstall the respective package (see `Issue#7 <https://github.com/rusty1s/pytorch_scatter/issues/7>`_). On macOS, it may help to install clang compilers via conda (see `Issue#18 <https://github.com/rusty1s/pytorch_geometric/issues/18>`_):

   .. code-block:: none

      $ conda install -y clang_osx-64 clangxx_osx-64 gfortran_osx-64

#. Unable to import ``*_cuda``: You need to ``import torch`` first before importing any of the extension packages (see `Issue#6 <https://github.com/rusty1s/pytorch_scatter/issues/6>`_).

#. ``error: command '/usr/bin/nvcc' failed with exit status 2``: Ensure that at least CUDA >= 8 is installed (see `Issue#25a <https://github.com/rusty1s/pytorch_geometric/issues/25>`_ and `Issue#106 <https://github.com/rusty1s/pytorch_geometric/issues/106>`_).

#. ``Segmentation fault (core dumped)``` or ``return __and_<is_constructible<_Elements, _UElements&&>...>::value``: Ensure that your ``gcc`` version is at least 4.9 and below 6 (see `Issue#25b <https://github.com/rusty1s/pytorch_scatter/issues/25>`_).
   You will also need to reinstall PyTorch because ``gcc`` versions must be consistent across all PyTorch packages.

#. ``undefined symbol: __cudaPopCallConfiguration``: Ensure that your PyTorch CUDA version and system CUDA version match (see `Issue#19 <https://github.com/rusty1s/pytorch_scatter/issues/19>`_):

   .. code-block:: none

      $ python -c "import torch; print(torch.version.cuda)"
      $ nvcc --version

#. On macOS: ``'gcc' failed with exit status 1``: Install the respective packages by using the following environment variables (see `Issue#21 <https://github.com/rusty1s/pytorch_scatter/issues/21>`_):

   .. code-block:: none

       $ MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

#. On macOS: ``ld: warning: directory not found for option '-L/usr/local/cuda/lib64'`` and ``ld: library not found for -lcudart``: Symlink ``cuda/lib`` to ``cuda/lib64`` (see `Issue#116 <https://github.com/rusty1s/pytorch_geometric/issues/116>`_):

   .. code-block:: none

       $ sudo ln -s /usr/local/cuda/lib /usr/local/cuda/lib64

#. On macOS: ``The version of the host compiler ('Apple clang') is not supported``: Downgrade your command line tools (see this `StackOverflow thread <https://stackoverflow.com/questions/36250949/revert-apple-clang-version-for-nvcc/46574116>`_) with the respective version annotated in the `CUDA Installation Guide for Mac <https://developer.download.nvidia.com/compute/cuda/10.1/Prod/docs/sidebar/CUDA_Installation_Guide_Mac.pdf>`_ (Section 1.1) for your specific CUDA version.
   You can download previous command line tool versions `here <https://idmsa.apple.com/IDMSWebAuth/signin?appIdKey=891bd3417a7776362562d2197f89480a8547b108fd934911bcbea0110d07f757&path=%2Fdownload%2Fmore%2F&rv=1>`_.
