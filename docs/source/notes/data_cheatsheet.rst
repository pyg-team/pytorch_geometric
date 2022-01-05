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
    * - :class:`~torch_geometric.datasets.{{ cls }}` (`Paper <http://dawdawd.com>`__)
      - 1
      - 2
      - 3
      - 4
      - 5
      - 6
{% endfor %}
