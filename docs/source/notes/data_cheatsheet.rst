Dataset Cheatsheet
==================

.. list-table::
    :widths: 50 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes/#tasks
{% for cls in torch_geometric.datasets.classes %}
    * - :class:`~torch_geometric.datasets.{{ cls }}` {% if torch_geometric.datasets.utils.paper_link(cls) %}(`Paper <{{ torch_geometric.datasets.utils.paper_link(cls) }}>`__){% endif %}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#graphs', default=1) }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#nodes', default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#edges', default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#features', default='') }}
      - dawd
    {% for child in torch_geometric.datasets.utils.get_children(cls) %}
    * - {{ child }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#graphs', child, default=1) }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#nodes', child, default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#edges', child, default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#features', child, default='') }}
      - dawd
    {% endfor %}
{% endfor %}
