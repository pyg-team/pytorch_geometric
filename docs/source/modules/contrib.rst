torch_geometric.contrib
=======================

.. currentmodule:: torch_geometric.contrib

.. warning::

    This module contains experimental code, which is not guaranteed to be
    stable. And the API may change in the future.

.. contents:: Contents
    :local:

GNNs
----

.. currentmodule:: torch_geometric.contrib.nn

.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.nn.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: torch_geometric.contrib.nn
   :members:
   :undoc-members:
   :exclude-members: message, aggregate, message_and_aggregate, update, MessagePassing, training, initialize_parameters

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
==========

.. currentmodule:: torch_geometric.contrib.transforms

.. autosummary::
    :nosignatures:
    {% for cls in torch_geometric.contrib.transforms.classes %}
      {{ cls }}
    {% endfor %}

.. automodule:: torch_geometric.contrib.transforms
    :members:
    :undoc-members:
