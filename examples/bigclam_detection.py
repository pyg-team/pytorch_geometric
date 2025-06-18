import os.path as osp

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import BigClam

# Cora
dataset = Planetoid(
    root=osp.join(osp.dirname(__file__), "data", "Planetoid"),
    name="Cora",
    transform=None,
)

data = dataset[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
data = data.to(device)
model = BigClam(
    data.edge_index,
    num_communities=7,
    device=device,
).to(device)


@torch.no_grad()
def main():

    model.fit(iterations=100, lr=0.005, verbose=True)

    model.eval()
    emb = model()

    hard_assignments = emb.argmax(dim=1).cpu().detach().numpy()

    # quantitative evaluation
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    nmi = normalized_mutual_info_score(data.y.cpu().numpy(), hard_assignments)
    ari = adjusted_rand_score(data.y.cpu().numpy(), hard_assignments)
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}")


main()
