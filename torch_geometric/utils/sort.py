import torch
import torch_geometric

from typing import Optional, Tuple

if torch_geometric.typing.WITH_PYG_LIB:  # pragma: no cover

    import pyg_lib

    def index_sort(
            inputs: torch.Tensor, max_value: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This function should be used only for positive integer values. See
        pyg-lib docuemntation for more details:
        https://pyg-lib.readthedocs.io/en/latest/modules/ops.html"""
        if inputs.dim() == 1 and is_integral(inputs):
            return pyg_lib.ops.index_sort(inputs, max_value=max_value)
        return inputs.sort()

else:

    def index_sort(
            inputs: torch.Tensor, max_value: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs.sort()


def is_integral(tensor: torch.Tensor) -> bool:
    return tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
