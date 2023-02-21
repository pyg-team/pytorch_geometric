{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objname != "MessagePassing" %}
.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :exclude-members: forward, reset_parameters, message, message_and_aggregate, edge_update, aggregate, update

   .. automethod:: forward
   .. automethod:: reset_parameters
{% else %}
.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
{% endif %}
