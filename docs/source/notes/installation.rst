Installation
============

We have outsourced a lot of functionality of PyTorch Geometric to other packages, which needs to be installed first.
These packages come with their own CPU and GPU kernel implementations based on `C FFI <https://github.com/pytorch/extension-ffi/>`_ and the newly introduced `C++/CUDA extensions <https://github.com/pytorch/extension-cpp/>`_.

If CUDA is available, ensure that ``nvcc`` is accessible from your terminal, e.g. by typing ``nvcc --version``. If not, add CUDA (e.g. ``/usr/local/cuda/bin``) to your ``$PATH``. Then run:

.. code-block:: none

    $ pip install cffi
    $ pip install --upgrade torch-scatter torch-unique torch-cluster torch-spline-conv
    $ pip install torch-geometric

Installation should be straightforward.
However, in rare cases CUDA or python path issues can prevent a succesful installation.

The error messages of ``pip`` in case of installation failure are unfortunately not very meaningful.
Therefore, you should clone the respective package and check where the error occurs:

.. code-block:: none

    $ git clone https://github.com/rusty1s/pytorch_scatter
    $ cd pytorch_scatter
    $ ./build.sh  # Check for CUDA compilation error.
    $ python setup.py install  # Check for link error.
    $ python setup.py test  # Verify.

Common installation issues
--------------------------

.. code-block:: none

    [Errno 13] Permission denied:

We do not recommend installation as root user on your system python.
Please setup an `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment or `Docker image <https://www.docker.com/>`_.

.. code-block:: none

     #include "cuda.h"
                      ^
    compilation terminated.

PyTorch can not find your CUDA.
Easily fixable by running:

.. code-block:: none

    $ CPATH=/usr/local/cuda/include python setup.py install
