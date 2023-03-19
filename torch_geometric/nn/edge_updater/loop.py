import inspect
from typing import Optional, Union, Dict, Callable

import torch
from torch import Tensor, LongTensor, BoolTensor
from torch_scatter import scatter

from torch_geometric.nn.edge_updater import EdgeUpdater
from torch_geometric.typing import OptTensor


def generate_mask(
        callback: Callable[..., BoolTensor],
        edge_index_i: LongTensor,
        edge_index_j: LongTensor,
        edge_weight: Tensor,
        edge_attr: OptTensor,
        edge_type: Tensor,
        **kwargs
) -> BoolTensor:
    kwargs['edge_index_i'] = edge_index_i
    kwargs['edge_index_j'] = edge_index_j
    kwargs['edge_weight'] = edge_weight
    kwargs['edge_attr'] = edge_attr
    kwargs['edge_type'] = edge_type

    params = {}
    for key, param in inspect.signature(callback).parameters.items():
        data = kwargs.get(key, inspect.Parameter.empty)
        if data is inspect.Parameter.empty:
            if param.default is inspect.Parameter.empty:
                raise TypeError(f'Required parameter {key} is empty.')
            data = param.default
        params[key] = data

    return callback(**params)


class MaskEdges(EdgeUpdater):
    r"""TODO
    """
    def forward(self, callback: Callable[..., BoolTensor],
                edge_index_i: LongTensor, edge_index_j: LongTensor,
                edge_weight: Tensor, edge_attr: OptTensor,
                edge_type: Tensor, **kwargs) -> Dict:
        mask = generate_mask(callback, edge_index_i, edge_index_j,
                             edge_weight, edge_attr, edge_type, **kwargs)
        assert mask.size() == edge_index_i.size()

        return {
            'edge_index_i': edge_index_i[mask],
            'edge_index_j': edge_index_j[mask],
            'edge_weight': edge_weight[mask],
            'edge_attr': None if edge_attr is None else edge_attr[mask],
            'edge_type': edge_type[mask],
        }


class RemoveSelfLoops(EdgeUpdater):
    def __init__(self):
        super().__init__()
        self._masker = MaskEdges()

    def forward(self, edge_index_i: LongTensor, edge_index_j: LongTensor,
                edge_weight: Tensor, edge_attr: OptTensor,
                edge_type: Tensor) -> Dict:
        def non_self_loops(
                edge_index_i: LongTensor,
                edge_index_j: LongTensor
        ) -> BoolTensor:
            return edge_index_i != edge_index_j

        return self._masker.forward(
            non_self_loops,
            edge_index_i, edge_index_j, edge_weight,
            edge_attr, edge_type
        )


class SegregateSelfLoops(EdgeUpdater):
    def __init__(self):
        super().__init__()
        self._masker = MaskEdges()

    def forward(self, edge_index_i: LongTensor, edge_index_j: LongTensor,
                edge_weight: Tensor, edge_attr: OptTensor,
                edge_type: Tensor) -> Dict:
        def self_loops(
                edge_index_i: LongTensor,
                edge_index_j: LongTensor
        ) -> BoolTensor:
            return edge_index_i == edge_index_j

        return self._masker.forward(
            self_loops,
            edge_index_i, edge_index_j, edge_weight,
            edge_attr, edge_type
        )


class AddMaskedSelfLoops(EdgeUpdater):
    r"""TODO"""
    def __init__(self,
                 fill_value: Optional[Union[float, Tensor, str]] = 1.0):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, callback: Callable[..., BoolTensor],
                edge_index_i: LongTensor, edge_index_j: LongTensor,
                edge_weight: Tensor, edge_attr: OptTensor,
                edge_type: Tensor, num_nodes: int,
                fill_value: Optional[Union[float, Tensor, str]] = None,
                **kwargs) -> Dict:
        mask = generate_mask(callback, edge_index_i, edge_index_j,
                             edge_weight, edge_attr, edge_type, **kwargs)
        assert mask.size() == edge_index_i.size()

        if fill_value is None:
            fill_value = self.fill_value

        def fill_tensor(input_tensor, fill_value, num_nodes, mask):
            if input_tensor is not None:
                if isinstance(fill_value, (int, float)):
                    loop_attr = input_tensor.new_full((num_nodes,) + input_tensor.size()[1:],
                                                      fill_value)
                elif isinstance(fill_value, Tensor):
                    loop_attr = fill_value.to(input_tensor.device, input_tensor.dtype)
                    if input_tensor.dim() != loop_attr.dim():
                        loop_attr = loop_attr.unsqueeze(0)
                    sizes = [num_nodes] + [1] * (loop_attr.dim() - 1)
                    loop_attr = loop_attr.repeat(*sizes)

                elif isinstance(fill_value, str):
                    loop_attr = scatter(input_tensor, edge_index_i, dim=0, dim_size=num_nodes,
                                        reduce=fill_value)
                else:
                    raise AttributeError("No valid 'fill_value' provided")

                inv_mask = ~mask
                loop_attr[edge_index_j[inv_mask]] = input_tensor[inv_mask]

                input_tensor = torch.cat([input_tensor[mask], loop_attr], dim=0)

            return input_tensor

        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index_i.device)

        return {
            'edge_index_i': torch.cat([edge_index_i[mask], loop_index], dim=0),
            'edge_index_j': torch.cat([edge_index_j[mask], loop_index], dim=0),
            'edge_weight': fill_tensor(edge_weight, fill_value, num_nodes, mask),
            'edge_attr': fill_tensor(edge_attr, fill_value, num_nodes, mask),
            'edge_type': fill_tensor(edge_type, fill_value, num_nodes, mask),
        }


class AddSelfLoops(EdgeUpdater):
    r"""TODO"""
    def __init__(self,
                 fill_value: Optional[Union[float, Tensor, str]] = 1.0):
        super().__init__()
        self._masker = AddMaskedSelfLoops(fill_value)

    def forward(self, edge_index_i: LongTensor, edge_index_j: LongTensor,
                edge_weight: Tensor, edge_attr: OptTensor, edge_type: Tensor,
                num_nodes: int,
                fill_value: Optional[Union[float, Tensor, str]] = None) -> Dict:
        def non_self_loops(edge_index_i) -> BoolTensor:
            return torch.ones_like(edge_index_i).bool()

        return self._masker.forward(non_self_loops, edge_index_i, edge_index_j,
                edge_weight, edge_attr, edge_type, num_nodes, fill_value)


class AddRemainingSelfLoops(EdgeUpdater):
    r"""TODO"""
    def __init__(self,
                 fill_value: Optional[Union[float, Tensor, str]] = 1.0):
        super().__init__()
        self._masker = AddMaskedSelfLoops(fill_value)

    def forward(self, edge_index_i: LongTensor, edge_index_j: LongTensor,
                edge_weight: Tensor, edge_attr: OptTensor, edge_type: Tensor,
                num_nodes: int,
                fill_value: Optional[Union[float, Tensor, str]] = None) -> Dict:
        def non_self_loops(
                edge_index_i: LongTensor,
                edge_index_j: LongTensor
        ) -> BoolTensor:
            return edge_index_i != edge_index_j

        return self._masker.forward(non_self_loops, edge_index_i, edge_index_j,
                edge_weight, edge_attr, edge_type, num_nodes, fill_value)


if __name__ == '__main__':  # TODO REMOVE
    edge_index = torch.tensor([
        [1, 1], [2, 1], [2, 2]
    ]).t()
    edge_index_i, edge_index_j = edge_index
    # edge_weight = torch.arange(3)
    edge_weight = torch.ones(3)
    edge_attr = torch.arange(3)
    edge_type = torch.arange(3)

    print(edge_index)
    print(edge_weight)

    # updater = RemoveSelfLoops()
    # updater = SegregateSelfLoops()
    updater = AddSelfLoops(fill_value='sum')
    updater = AddRemainingSelfLoops(fill_value=-1)
    x = updater(
        edge_index_i=edge_index_i,
        edge_index_j=edge_index_j,
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        edge_type=edge_type,
        num_nodes=3,
    )
    print(x)

