torch_geometric.contrib
=======================

.. currentmodule:: torch_geometric.contrib

:obj:`torch_geometric.contrib` is a staging area for early stage experimental code.
Modules might be moved to the main library in the future.

.. warning::

    This module contains experimental code, which is not guaranteed to be stable.

.. contents:: Contents
    :local:

Convolutional Layers
--------------------

.. currentmodule:: torch_geometric.contrib.nn.conv

.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.contrib.nn.conv.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: torch_geometric.contrib.nn.conv
   :members:
   :undoc-members:
   :exclude-members: message, aggregate, message_and_aggregate, update, MessagePassing, training, initialize_parameters

Models
------

Relational prediction
~~~~~~~~~~~~~~~~~~~~~

:class:`~torch_geometric.contrib.nn.models.RelGTTokenEncoder` combines row
embeddings with table, hop, time, and structural encodings.
:class:`~torch_geometric.contrib.nn.models.RelGT` applies local attention and
persistent centroid-based global attention to predict seed-level scores.
The models accept tensors without requiring RelBench or TorchFrame.
The `RelGT example <https://github.com/pyg-team/pytorch_geometric/tree/master/examples/contrib/relgt>`_
shows how to build prediction-time contexts, train on simulated customer
histories, and map a RelBench task row to a seed and cutoff.

.. currentmodule:: torch_geometric.contrib.nn.models

.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.contrib.nn.models.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: torch_geometric.contrib.nn.models
   :members:
   :undoc-members:
   :exclude-members: message, aggregate, message_and_aggregate, update, MessagePassing, training, init_conv

Datasets
--------

.. currentmodule:: torch_geometric.contrib.datasets

.. autosummary::
    :nosignatures:
    {% for cls in torch_geometric.contrib.datasets.classes %}
      {{ cls }}
    {% endfor %}

.. automodule:: torch_geometric.contrib.datasets
    :members:
    :exclude-members: download, process, processed_file_names, raw_file_names, num_classes, get

Transforms
----------

.. currentmodule:: torch_geometric.contrib.transforms

.. autosummary::
    :nosignatures:
    {% for cls in torch_geometric.contrib.transforms.classes %}
      {{ cls }}
    {% endfor %}

.. automodule:: torch_geometric.contrib.transforms
    :members:


Explainer
---------

.. currentmodule:: torch_geometric.contrib.explain

.. autosummary::
    :nosignatures:
    {% for cls in torch_geometric.contrib.explain.classes %}
      {{ cls }}
    {% endfor %}

.. automodule:: torch_geometric.contrib.explain
    :members:
