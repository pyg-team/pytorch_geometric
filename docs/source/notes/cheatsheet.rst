GNN Cheatsheet
==============

Graph Neural Network Operators
------------------------------

.. list-table::
    :widths: 60 10 10 10 10
    :header-rows: 1

    * - Name
      - :class:`~torch_sparse.SparseTensor`
      - :obj:`edge_weight`
      - :obj:`edge_attr`
      - bipartite
{% for cls in torch_geometric.nn.conv.classes[1:] %}
{% if not torch_geometric.nn.conv.utils.processes_heterogeneous_graphs(cls) and
      not torch_geometric.nn.conv.utils.processes_hypergraphs(cls) and
      not torch_geometric.nn.conv.utils.processes_point_clouds(cls) %}
    * - :class:`~torch_geometric.nn.conv.{{ cls }}` (`Paper <{{ torch_geometric.nn.conv.utils.paper_link(cls) }}>`__)
      - {% if torch_geometric.nn.conv.utils.supports_sparse_tensor(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_edge_weights(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_edge_features(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_bipartite_graphs(cls) %}✓{% endif %}
{% endif %}
{% endfor %}

Heterogeneous Graph Neural Network Operators
--------------------------------------------

.. list-table::
    :widths: 60 10 10 10 10
    :header-rows: 1

    * - Name
      - :class:`~torch_sparse.SparseTensor`
      - :obj:`edge_weight`
      - :obj:`edge_attr`
      - bipartite
{% for cls in torch_geometric.nn.conv.classes[1:] %}
{% if torch_geometric.nn.conv.utils.processes_heterogeneous_graphs(cls) %}
    * - :class:`~torch_geometric.nn.conv.{{ cls }}` (`Paper <{{ torch_geometric.nn.conv.utils.paper_link(cls) }}>`__)
      - {% if torch_geometric.nn.conv.utils.supports_sparse_tensor(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_edge_weights(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_edge_features(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_bipartite_graphs(cls) %}✓{% endif %}
{% endif %}
{% endfor %}

Hypergraph Neural Network Operators
-----------------------------------

.. list-table::
    :widths: 60 10 10 10 10
    :header-rows: 1

    * - Name
      - :class:`~torch_sparse.SparseTensor`
      - :obj:`edge_weight`
      - :obj:`edge_attr`
      - bipartite
{% for cls in torch_geometric.nn.conv.classes[1:] %}
{% if torch_geometric.nn.conv.utils.processes_hypergraphs(cls) %}
    * - :class:`~torch_geometric.nn.conv.{{ cls }}` (`Paper <{{ torch_geometric.nn.conv.utils.paper_link(cls) }}>`__)
      - {% if torch_geometric.nn.conv.utils.supports_sparse_tensor(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_edge_weights(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_edge_features(cls) %}✓{% endif %}
      - {% if torch_geometric.nn.conv.utils.supports_bipartite_graphs(cls) %}✓{% endif %}
{% endif %}
{% endfor %}

Point Cloud Neural Network Operators
------------------------------------

.. list-table::
    :widths: 90 10
    :header-rows: 1

    * - Name
      - bipartite
{% for cls in torch_geometric.nn.conv.classes[1:] %}
{% if torch_geometric.nn.conv.utils.processes_point_clouds(cls) %}
    * - :class:`~torch_geometric.nn.conv.{{ cls }}` (`Paper <{{ torch_geometric.nn.conv.utils.paper_link(cls) }}>`__)
      - {% if torch_geometric.nn.conv.utils.supports_bipartite_graphs(cls) %}✓{% endif %}
{% endif %}
{% endfor %}
