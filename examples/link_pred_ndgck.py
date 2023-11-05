import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, to_dense_adj

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class NdcgK():
    def __init__(self, k: int):
        self.k = k
        self.dcg = torch.Tensor([])
        self.idcg = torch.Tensor([])

    def _dcg(self, pred_mat: torch.Tensor,
             label_mat: torch.Tensor) -> torch.Tensor:
        """Computes the Discounted cumulative gain for given link-predictions
        and valid links.

        Args:
            pred_mat (torch.Tensor): The link-predictions matrix.
            label_mat (torch.Tensor): The valid link matrix.

        Returns:
            torch.Tensor: The DCG value per head node.
        """

        top_k_pred_mat = pred_mat.argsort(dim=1, descending=True)[:, :self.k]
        pred_mat_heads = torch.arange(top_k_pred_mat.size(0)).repeat(
            self.k, 1).T.reshape(-1)
        top_k_pred_labels = label_mat[(
            pred_mat_heads,
            top_k_pred_mat.reshape(-1))].reshape_as(top_k_pred_mat)
        log_ranks = torch.log2(torch.arange(self.k) + 2)
        return (top_k_pred_labels / log_ranks).sum(dim=1)

    def update(self, pred_mat: torch.Tensor,
               pos_edge_label_index: torch.Tensor):
        """Updates state with predictions and targets.

        Args:
            pred_mat (torch.Tensor): The matrix containing link-predictions
                for each pair of nodes.
            pos_edge_label_index (torch.Tensor): The edge indices of
                valid links.
        """
        assert pred_mat.size(
            1) >= self.k, f"At least {self.k} entries for each query."

        edge_label_matrix = to_dense_adj(pos_edge_label_index,
                                         max_num_nodes=pred_mat.size(0)).view(
                                             pred_mat.size(0), -1)

        row_dcg = self._dcg(pred_mat, edge_label_matrix)
        row_idcg = self._dcg(edge_label_matrix, edge_label_matrix)

        self.dcg = torch.concat((self.dcg, row_dcg))
        self.idcg = torch.concat((self.idcg, row_idcg))

    def compute(self) -> torch.Tensor:
        """Computes NDCG@k based on previous updates.

        Returns:
             torch.Tensor: The average NDCG@k value.
        """
        non_zeros = self.idcg.nonzero()
        return (self.dcg[non_zeros] / self.idcg[non_zeros]).mean()


transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora', transform=transform)
# After applying the `RandomLinkSplit` transform, the data is transformed from
# a data object to a list of tuples (train_data, val_data, test_data), with
# each element representing the corresponding split.
train_data, val_data, test_data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return prob_adj


model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
ndcgk = NdcgK(k=10)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    pred_mat = model.decode_all(z)
    ndcgk.update(pred_mat, test_data.edge_label_index)
    return ndcgk.compute()


best_val_ndcg = final_test_ndcg = 0
for epoch in range(1, 101):
    loss = train()
    val_ndcg = test(val_data)
    test_ndcg = test(test_data)
    if val_ndcg > best_val_ndcg:
        best_val_ndcg = val_ndcg
        final_test_ndcg = test_ndcg
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_ndcg:.4f}, '
          f'Test: {test_ndcg:.4f}')

print(f'Final Test: {final_test_ndcg:.4f}')
