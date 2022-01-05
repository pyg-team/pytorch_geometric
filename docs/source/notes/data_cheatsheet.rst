Dataset Cheatsheet
==================

.. list-table::
    :widths: 30 10 10 10 10 20 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - task
      - hetero
{% for cls in torch_geometric.datasets.classes %}
    * - :class:`~torch_geometric.datasets.{{ cls }}` (`Paper <http://foo.com>`__)
      -
      -
      -
      -
      -
      -
{% endfor %}
