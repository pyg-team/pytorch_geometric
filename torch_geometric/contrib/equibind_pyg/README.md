# EquiBind-PyG (Rigid Docking)

This contrib module provides a PyTorch Geometric implementation of the **rigid EquiBind protein–ligand docking model**, adapted for modularity, readability, and future extension toward flexible docking.

## Features

- **Rigid EquiBind model** (`EquiBindRigid`)
- **EGNN** and **IEGNN** message-passing layers
- **Keypoint attention mechanism**
- **Kabsch alignment** and geometric utilities
- **RMSD / Kabsch RMSD / Centroid metrics**
- **Loss functions** used in rigid docking
- Fully self-contained, no training scripts required

## Usage

```python
from torch_geometric.contrib.equibind_pyg import EquiBindRigid

model = EquiBindRigid(
    node_dim=128,
    num_layers=4,
    num_keypoints=32,
    edge_mlp_dim=64,
    coord_mlp_dim=64,
    cross_attn_dim=64,
)

All layers and geometry utilities are available under:

torch_geometric.contrib.equibind_pyg.layers
torch_geometric.contrib.equibind_pyg.models
torch_geometric.contrib.equibind_pyg.geometry

