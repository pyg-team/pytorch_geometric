:orphan:

Dataset Cheatsheet
==================

.. note::

    This dataset statistics table is a **work in progress**.
    Please consider helping us filling its content by providing statistics for individual datasets.
    See `here <https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/karate.py#L25-L37>`__ and `here <https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/tu_dataset.py#L56-L108>`__ for examples on how to do so.

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
      - {%if torch_geometric.datasets.utils.has_stats(cls) %}{{ torch_geometric.datasets.utils.get_stat(cls, '#graphs', default=1) }}{% else %}{{ torch_geometric.datasets.utils.get_stat(cls, '#graphs', default='') }}{% endif %}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#nodes', default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#edges', default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#features', default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#classes', default='') }}{{ torch_geometric.datasets.utils.get_stat(cls, '#tasks', default='') }}
    {% for child in torch_geometric.datasets.utils.get_children(cls) %}
    * - └─ {{ child }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#graphs', child, default=1) }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#nodes', child, default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#edges', child, default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#features', child, default='') }}
      - {{ torch_geometric.datasets.utils.get_stat(cls, '#classes', child, default='') }}{{ torch_geometric.datasets.utils.get_stat(cls, '#tasks', child, default='') }}
    {% endfor %}
{% endfor %}
