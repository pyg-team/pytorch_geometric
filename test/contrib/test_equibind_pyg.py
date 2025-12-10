import torch
from torch_geometric.data import HeteroData
from torch_geometric.contrib.equibind_pyg import EquiBindRigid


def test_equibind_rigid_forward_shapes():
    # Tiny dummy HeteroData that matches model API
    data = HeteroData()

    # 4 ligand atoms, 6 receptor residues
    data["ligand"].x = torch.randn(4, 28)
    data["ligand"].pos = torch.randn(4, 3)
    data["ligand", "intra", "ligand"].edge_index = torch.tensor(
        [[0, 1, 2, 3],
         [1, 0, 3, 2]],
        dtype=torch.long,
    )

    data["receptor"].x = torch.randn(6, 21)
    data["receptor"].pos = torch.randn(6, 3)
    data["receptor", "intra", "receptor"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5],
         [1, 0, 3, 2, 5, 4]],
        dtype=torch.long,
    )

    # Ground truth ligand positions needed by the loss
    data.ligand_pos_bound = data["ligand"].pos.clone()

    model = EquiBindRigid(
        node_dim=128,
        num_layers=2,
        num_keypoints=4,
        edge_mlp_dim=32,
        coord_mlp_dim=32,
        cross_attn_dim=32,
        lig_in_dim=28,
        rec_in_dim=21,
    )

    out = model(data)
    assert "ligand_pos_pred" in out
    assert out["ligand_pos_pred"].shape == data["ligand"].pos.shape
