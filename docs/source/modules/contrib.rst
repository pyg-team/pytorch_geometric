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
    :undoc-members:
