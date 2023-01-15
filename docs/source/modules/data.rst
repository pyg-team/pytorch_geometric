torch_geometric.data
====================

.. contents:: Contents
    :local:

Data Objects
------------

.. currentmodule:: torch_geometric.data

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   Data
   HeteroData
   Batch
   TemporalData
   Dataset
   InMemoryDataset

Remote Backend Interfaces
-------------------------

.. currentmodule:: torch_geometric.data

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   FeatureStore
   GraphStore
   TensorAttr
   EdgeAttr

PyTorch Lightning Wrappers
--------------------------

.. currentmodule:: torch_geometric.data.lightning

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   LightningDataset
   LightningNodeData
   LightningLinkData

Helper Functions
----------------

.. currentmodule:: torch_geometric.data

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   makedirs
   download_url
   extract_tar
   extract_zip
   extract_bz2
   extract_gz
