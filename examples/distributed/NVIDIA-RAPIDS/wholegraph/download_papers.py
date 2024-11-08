from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

transform = T.Compose([T.ToUndirected(), T.ToSparseTensor()])
dataset = PygNodePropPredDataset(name='ogbn-papers100M',
                                 root='/workspace',
                                 pre_transform=transform)