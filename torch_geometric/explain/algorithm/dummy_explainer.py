from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import ExplainerConfig, ModelConfig


class DummyExplainer(ExplainerAlgorithm):
    r"""A dummy explainer for testing purposes."""
    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None, **kwargs) -> Explanation:
        r"""Returns random explanations based on the shape of the inputs.

        Args:
            x (torch.Tensor): node features.
            edge_index (torch.Tensor): edge indices.
            edge_attr (torch.Tensor, optional): edge attributes.
                Defaults to None.

        Returns:
            Explanation: A random explanation based on the shape of the inputs.
        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        mask_dict = {}
        mask_dict['node_feat_mask'] = torch.rand_like(x)
        mask_dict['node_mask'] = torch.rand(num_nodes, device=x.device)
        mask_dict['edge_mask'] = torch.rand(num_edges, device=x.device)

        if edge_attr is not None:
            mask_dict["edge_features_mask"] = torch.rand_like(edge_attr)

        return Explanation(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            **mask_dict,
        )

    def loss(self, y_hat: Tensor, y: Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y)

    def supports(
        self,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> bool:
        return True
