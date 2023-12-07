import argparse
import os.path as osp

import torch
import torch.optim as optim
from transh import TransH

from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import TransE
from torch_geometric.transforms import RandomLinkSplit

model_map = {'transe': TransE, 'transh': TransH}

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=model_map.keys(), type=str.lower,
                    required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FB15k')

train_data = FB15k_237(path, split='train')[0].to(device)
print('train_data size:', train_data.size)
val_data = FB15k_237(path, split='val')[0].to(device)
print('val_data size:', val_data.size)
test_data = FB15k_237(path, split='test')[0].to(device)

# # our sanity check (DELETE LATER)
# transform = RandomLinkSplit(is_undirected=True)
# train_test, val_test, test_test = transform(test_data)
# #transform2 = RandomLinkSplit(is_undirected=True)
# #train_test2, val_test2, test_test2 = transform2(test_test)
# test_data = val_test
# print('test data:', val_test.size)

model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args.model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
    **model_arg_map.get(args.model, {}),
).to(device)

loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
)

optimizer_map = {
    'transe': optim.Adam(model.parameters(), lr=0.01),
    'transh': optim.Adam(model.parameters(), lr=0.01)
    # 'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    # 'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    # 'rotate': optim.Adam(model.parameters(), lr=1e-3),
}
optimizer = optimizer_map[args.model]


def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=1000,  #from 20000
        k=10,
    )


model.compute_corrupt_probs(train_data.edge_index[0], train_data.edge_type,
                            train_data.edge_index[1])
for epoch in range(1, 500):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 25 == 0:
        rank, mrr, hits = test(val_data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

rank, mrr, hits_at_10 = test(test_data)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')
"""
TransH initial test:
- mean rank: 7592.13
- MRR: 0.0076
- hits@10: 0.0123

TransE initial test:
- mean rank: 7222.76
- MRR: 0.0408
- hits@10: 0.0737
"""
