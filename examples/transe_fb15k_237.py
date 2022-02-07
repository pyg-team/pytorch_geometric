import os.path as osp

import torch

from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import TransE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FB15k')
train_data = FB15k_237(path, split='train')[0].to(device)
val_data = FB15k_237(path, split='val')[0].to(device)
test_data = FB15k_237(path, split='test')[0].to(device)

model = TransE(train_data.num_nodes, train_data.num_edge_types,
               hidden_channels=50, p_norm=1.0, sparse=True).to(device)

loader = model.loader(train_data.edge_index, train_data.edge_type,
                      batch_size=10000, shuffle=True, drop_last=True)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

for epoch in range(1, 1001):
    total_loss = total_examples = 0
    for edge_index, edge_type, pos_mask in loader:
        optimizer.zero_grad()
        loss = model.loss(edge_index, edge_type, pos_mask)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * edge_index.size(1)
        total_examples += edge_index.size(1)
    loss = total_loss / total_examples
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}')

    if epoch % 100 == 0:
        rank, hits = model.test(val_data.edge_index, val_data.edge_type,
                                batch_size=20000, k=10)
        print(f'Epoch: {epoch:04d}, Rank: {rank:.2f}, Hits: {hits:.4f}')

rank, hits = model.test(test_data.edge_index, test_data.edge_type,
                        batch_size=20000, k=10)
print(f'Rank: {rank:.2f}, Hits: {hits:.4f}')
