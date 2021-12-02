import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from torch_geometric.nn import LabelPropagation

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OGB')
dataset = PygNodePropPredDataset(
    'ogbn-arxiv', root, transform=T.Compose([
        T.ToUndirected(),
        T.ToSparseTensor(),
    ]))
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-arxiv')
data = dataset[0]

model = LabelPropagation(num_layers=3, alpha=0.9)
out = model(data.y, data.adj_t, mask=split_idx['train'])

y_pred = out.argmax(dim=-1, keepdim=True)

val_acc = evaluator.eval({
    'y_true': data.y[split_idx['valid']],
    'y_pred': y_pred[split_idx['valid']],
})['acc']
test_acc = evaluator.eval({
    'y_true': data.y[split_idx['test']],
    'y_pred': y_pred[split_idx['test']],
})['acc']

print(f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
