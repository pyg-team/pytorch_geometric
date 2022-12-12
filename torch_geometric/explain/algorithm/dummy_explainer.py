from collections import defaultdict
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.typing import EdgeType, NodeType


class DummyExplainer(ExplainerAlgorithm):
    r"""A dummy explainer for testing purposes."""
    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        edge_attr: Optional[Union[Tensor, Dict[EdgeType, Tensor]]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        r"""Returns random explanations based on the shape of the inputs.

        Args:
            model (torch.nn.Module): The model to explain.
            x (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The input
                node features of a homogeneous or heterogeneous graph.
            edge_index (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The
                input edge indices of a homogeneous or heterogeneous graph.
            edge_attr (Union[torch.Tensor, Dict[EdgeType, torch.Tensor]],
                optional): The inputs edge attributes of a homogeneous or
                heterogeneous graph. (default: :obj:`None`)

        Returns:
            Union[Explanation, HeteroExplanation]: A random explanation based
                on the shape of the inputs.
        """
        assert isinstance(x, (Tensor, dict))

        if isinstance(x, Tensor):
            assert isinstance(edge_index, Tensor)

            return Explanation(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_mask=torch.rand(x.size(0), device=x.device),
                node_feat_mask=torch.rand_like(x),
                edge_mask=torch.rand(edge_index.size(1), device=x.device),
                edge_feat_mask=torch.rand_like(edge_attr)
                if edge_attr is not None else None,
            )
        else:
            assert isinstance(edge_index, dict)

            node_dict = defaultdict(dict)
            for key, x in x.items():
                node_dict[key]['x'] = x
                node_dict[key]['node_mask'] = torch.rand(
                    x.size(0), device=x.device)
                node_dict[key]['node_feat_mask'] = torch.rand_like(x)

            edge_dict = defaultdict(dict)
            for key, edge_index in edge_index.items():
                edge_dict[key]['edge_index'] = edge_index
                edge_dict[key]['edge_mask'] = torch.rand(
                    edge_index.size(1), device=edge_index.device)
                if edge_attr is not None:
                    edge_dict[key]['edge_attr'] = edge_attr[key]
                    edge_dict[key]['edge_feat_mask'] = torch.rand_like(
                        edge_attr[key])

            return HeteroExplanation({**node_dict, **edge_dict})

    def supports(self) -> bool:
        return True
