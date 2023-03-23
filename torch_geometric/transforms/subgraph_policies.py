from typing import Any, Callable, List, Optional

import torch

from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected
from torch_geometric.utils.dropout import filter_adj


class SubgraphData(Data):
    """ A data object describing a collection of subgraphs generated from
  a subgraph selection policy. It has several additional (**) properties:

  From Data
  * x
  * edge_index
  * edge_attr
  * y
  * pos

  Additional
  ** :obj:`subgraph_id` (Tensor): The indices of the subgraphs
  ** :obj:`subgraph_batch` (Tensor): The batch vector of the subgraphs
  ** :obj:`subgraph_n_id` (Tensor): The indices of nodes in the subgraphs
  ** :obj:`orig_edge_index` (Tensor): The edge index of the original graph
  ** :obj:`orig_edge_attr` (Tensor): The edge attribute of the original graph
  ** :obj:`num_subgraphs` (int): The number of generated subgraphs
  ** :obj:`num_nodes_per_subgraph` (int): The number of nodes in the graph
  """
    def __inc__(self, key, value, *args, **kwargs) -> Any:
        if key == 'orig_edge_index':
            return self.num_nodes_per_subgraph
        elif key == 'subgraph_batch':
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)


class SubgraphPolicy(BaseTransform):
    def __init__(self, subgraph_transform: Optional[Callable] = None):
        self.subgraph_transform = subgraph_transform

    def graph_to_subgraphs(self, data: Data) -> List[Data]:
        raise NotImplementedError

    def __call__(self, data: Data) -> SubgraphData:
        assert data.is_undirected()

        subgraphs = self.graph_to_subgraphs(data)
        if self.subgraph_transform is not None:
            subgraphs = [
                self.subgraph_transform(subgraph) for subgraph in subgraphs
            ]
        subgraph_batch = Batch.from_data_list(subgraphs)

        # Batch subgraphs data
        out = SubgraphData(x=subgraph_batch.x, y=data.y,
                           edge_index=subgraph_batch.edge_index,
                           edge_attr=subgraph_batch.edge_attr,
                           subgraph_batch=subgraph_batch.batch,
                           subgraph_id=subgraph_batch.subgraph_id,
                           subgraph_n_id=subgraph_batch.subgraph_n_id,
                           orig_edge_index=data.edge_index,
                           orig_edge_attr=data.edge_attr,
                           num_nodes_per_subgraph=data.num_nodes,
                           num_subgraphs=len(subgraphs))

        return out


class NodeDeletionPolicy(SubgraphPolicy):
    def graph_to_subgraphs(self, data: Data) -> List[Data]:
        subgraphs = []
        num_nodes = data.num_nodes
        nodes_index = torch.arange(num_nodes)

        for i in range(num_nodes):
            subgraph_edge_index, subgraph_edge_attr = subgraph(
                subset=torch.cat([nodes_index[:i], nodes_index[i + 1:]]),
                edge_index=data.edge_index, edge_attr=data.edge_attr,
                num_nodes=num_nodes)
            subgraph_id = torch.tensor(i)
            subgraph_data = Data(x=data.x, edge_index=subgraph_edge_index,
                                 edge_attr=subgraph_edge_attr,
                                 subgraph_id=subgraph_id,
                                 subgraph_n_id=nodes_index,
                                 num_nodes=num_nodes)
            subgraphs.append(subgraph_data)

        return subgraphs


class EdgeDeletionPolicy(SubgraphPolicy):
    def graph_to_subgraphs(self, data) -> List[Data]:
        subgraphs = []
        num_nodes = data.num_nodes
        nodes_index = torch.arange(num_nodes)

        head, tail = data.edge_index

        # Handle the edge case of graph having no edges
        if data.num_edges == 0:
            subgraphs.append(
                Data(x=data.x, edge_index=data.edge_index,
                     edge_attr=data.edge_attr, subgraph_id=torch.tensor(0),
                     subgraph_n_id=nodes_index, num_nodes=num_nodes))
            return subgraphs

        # Calling the method from torch_geometric.utils.dropout
        head, tail, edge_attr = filter_adj(row=head, col=tail,
                                           edge_attr=data.edge_attr,
                                           mask=head < tail)

        for i in range(head.size(0)):
            subgraph_head = torch.cat([head[:i], head[i + 1:]])
            subgraph_tail = torch.cat([tail[:i], tail[i + 1:]])
            subgraph_edge_attr = torch.cat([
                edge_attr[:i], edge_attr[i + 1:]
            ]) if edge_attr is not None else edge_attr
            subgraph_edge_index = torch.stack([subgraph_head, subgraph_tail],
                                              dim=0)

            # The subgraph_edge_attr is None case is only
            # required for PyG < 2.3.0. Otherwise to_undirected
            # always returns a tuple
            if subgraph_edge_attr is None:
                subgraph_edge_index = to_undirected(
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr, num_nodes=num_nodes)
            else:
                subgraph_edge_index, subgraph_edge_attr = to_undirected(
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr, num_nodes=num_nodes)

            subgraph_id = torch.tensor(i)
            subgraph_data = Data(x=data.x, edge_index=subgraph_edge_index,
                                 edge_attr=subgraph_edge_attr,
                                 subgraph_id=subgraph_id,
                                 subgraph_n_id=nodes_index,
                                 num_nodes=num_nodes)
            subgraphs.append(subgraph_data)

        return subgraphs


# EGO and EGO+
class EgoNetPolicy(SubgraphPolicy):
    def __init__(self, num_hops: int,
                 subgraph_transform: Optional[Callable] = None,
                 ego_plus: Optional[bool] = False):
        super().__init__(subgraph_transform)
        self.ego_plus = ego_plus
        self.num_hops = num_hops

    def graph_to_subgraphs(
        self,
        data: Data,
    ) -> List[Data]:
        subgraphs = []
        subgraph_data = None
        num_nodes = data.num_nodes
        nodes_index = torch.arange(num_nodes)

        for i in range(num_nodes):
            subgraph_id = torch.tensor(i)
            _, subgraph_edge_index, _, edge_mask = k_hop_subgraph(
                i, self.num_hops, data.edge_index, num_nodes=num_nodes)
            subgraph_edge_attr = data.edge_attr[
                edge_mask] if data.edge_attr is not None else data.edge_attr
            subgraph_x = data.x

            # add node features for EGO+ policy
            if self.ego_plus:
                # for the central node i, prepend a feature [1, 0]
                # for all non-central nodes, prepend a feature [0, 1]

                prepend_features = torch.tensor(
                    [[0, 1] if j != i else [1, 0]
                     for j in range(num_nodes)], ).to(
                         subgraph_edge_index.device, torch.float)
                subgraph_x = torch.hstack([
                    prepend_features, subgraph_x
                ]) if subgraph_x is not None else prepend_features

            subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index,
                                 edge_attr=subgraph_edge_attr,
                                 subgraph_id=subgraph_id,
                                 subgraph_n_id=nodes_index,
                                 num_nodes=num_nodes)
            subgraphs.append(subgraph_data)

        return subgraphs


def subgraph_policy(policy: str, subgraph_transform: Optional[Callable] = None,
                    num_hops: Optional[int] = 0) -> SubgraphPolicy:
    if policy == 'node_deletion':
        return NodeDeletionPolicy(subgraph_transform=subgraph_transform)
    elif policy == 'edge_deletion':
        return EdgeDeletionPolicy(subgraph_transform=subgraph_transform)
    elif policy == 'ego':
        return EgoNetPolicy(num_hops=num_hops, ego_plus=False,
                            subgraph_transform=subgraph_transform)
    elif policy == 'ego_plus':
        return EgoNetPolicy(num_hops=num_hops, ego_plus=True,
                            subgraph_transform=subgraph_transform)
    elif policy == "original":
        return subgraph_transform

    raise ValueError("Subgraph policy not supported")
