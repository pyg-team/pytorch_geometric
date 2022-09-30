torch_geometric.sampler
=======================

.. currentmodule:: torch_geometric.sampler

.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.sampler.classes %}
     {{ cls }}
   {% endfor %}

.. autoclass:: torch_geometric.sampler.base.BaseSampler
   :members:

.. automodule:: torch_geometric.sampler
   :members:
   :undoc-members:
   :exclude-members: BaseSampler, sample_from_nodes, sample_from_edges, edge_permutation
