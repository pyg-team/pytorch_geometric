import functools
import math

from typing import Dict, Any, Tuple, Optional

import torch

from torch.nn import Module, Parameter
from torch.nn.init import trunc_normal_
from torch_geometric.data import Data

from torch_geometric.contrib.nn import GLTModel

EDGE_MASK = GLTModel.EDGE_MASK + GLTModel.MASK
INIT_FUNC = functools.partial(trunc_normal_, mean=1, a=1 - 1e-3, b=1 + 1e-3)


class GLTMask:
    def __init__(self, module: Module, graph: Data, device: torch.device, ignore_keys: Optional[set] = None) -> None:
        self.graph_mask = INIT_FUNC(
            torch.ones((graph.edge_index.shape[1] or graph.num_edges), device=device)
        )
        self.weight_mask = {
            param_name + GLTModel.MASK: INIT_FUNC(torch.ones_like(param))
            for param_name, param in module.named_parameters()
            if param_name not in ignore_keys
        }

    def sparsity(self) -> Tuple[float, float]:
        norm_graph_mask = float(torch.count_nonzero(self.graph_mask))
        norm_graph = torch.numel(self.graph_mask)
        graph_sparsity = 1 - norm_graph_mask / norm_graph

        norm_weight_mask = 0
        norm_weight = 0

        for v in self.weight_mask.values():
            norm_weight_mask += float(torch.count_nonzero(v))
            norm_weight += torch.numel(v)

        weight_sparsity = 1 - norm_weight_mask / norm_weight
        return graph_sparsity, weight_sparsity

    def to_dict(self, weight_prefix=False) -> Dict[str, Any]:
        pref = "module." if weight_prefix else ""

        return {
            EDGE_MASK: self.graph_mask.detach().clone(),
            **{pref + k: v.detach().clone() for k, v in self.weight_mask.items()},
        }

    def load_and_binarise(
            self,
            model_masks: Dict[str, Parameter],
            p_theta: float,
            p_g: float,
    ) -> None:
        # Validation
        missing_masks = [
            name
            for name in [
                EDGE_MASK,
                *self.weight_mask.keys(),
            ]
            if name not in model_masks.keys()
        ]

        if len(missing_masks):
            raise ValueError(
                f"Model has no masks for the following parameters: {missing_masks}"
            )

        # splitting out m_g and m_theta
        graph_mask = model_masks[EDGE_MASK]
        del model_masks[EDGE_MASK]

        # process graph mask
        self.graph_mask = torch.where(
            self.graph_mask > 0, 1.0, 0.
        )  # needed to support non-binary inits
        all_weights_graph = graph_mask[self.graph_mask == 1]
        num_prune_graph = min(
            math.floor(p_g * len(all_weights_graph)), len(all_weights_graph) - 1
        )
        threshold_graph = all_weights_graph.sort()[0][num_prune_graph]
        self.graph_mask = torch.where(
            graph_mask > threshold_graph, self.graph_mask, 0.
        )

        # process weight masks
        self.weight_mask = {
            k: torch.where(v > 0, 1.0, 0.0) for k, v in self.weight_mask.items()
        }  # needed to support non-binary inits
        all_weights_model = torch.concat(
            [v[self.weight_mask[k] == 1] for k, v in model_masks.items()]
        )
        num_prune_weights = min(
            math.floor(p_theta * len(all_weights_model)), len(all_weights_model) - 1
        )
        threshold_model = all_weights_model.sort()[0][num_prune_weights]

        self.weight_mask = {
            k: torch.where(v > threshold_model, self.weight_mask[k], 0.0)
            for k, v in model_masks.items()
        }
