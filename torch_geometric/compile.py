import logging
import warnings

try:
    from torch.utils._contextlib import _DecoratorContextManager
except ImportError:
    from torch.autograd.grad_mode import _DecoratorContextManager

import torch_geometric.typing


class enable_compile(_DecoratorContextManager):
    r"""Context-manager to enable :meth:`torch.compile` mode in :pyg:`PyG`.

    Specifically, this context-manager temporarily disables the usage of the
    external extension packages :obj:`pyg_lib`, :obj:`torch_scatter` and
    :obj:`torch_sparse` as they are not compatible with :meth:`torch.compile`.

    .. code-block:: python

        # Use as a context manager:
        with torch_geometric.enable_compile():
            model = torch.compile(model)

        # Use as a decorator:
        @torch_geometric.enable_compile()
        def main(model):
            model = torch.compile(model)

    Args:
        log_level (int, optional): The log level to use during
            :meth:`torch.compile`. (default: :obj:`logging.WARNING`)
    """
    def __init__(self, log_level: int = logging.WARNING):
        super().__init__()
        self.log_level = log_level

        self.prev_state = {}
        self.prev_log_level = {}

    def __enter__(self):
        self.prev_state = {
            'WITH_PYG_LIB': torch_geometric.typing.WITH_PYG_LIB,
            'WITH_SAMPLED_OP': torch_geometric.typing.WITH_SAMPLED_OP,
            'WITH_INDEX_SORT': torch_geometric.typing.WITH_INDEX_SORT,
            'WITH_TORCH_SCATTER': torch_geometric.typing.WITH_TORCH_SCATTER,
            'WITH_TORCH_SPARSE': torch_geometric.typing.WITH_TORCH_SPARSE,
        }
        self.prev_log_level = {
            'torch._dynamo': logging.getLogger('torch._dynamo').level,
            'torch._inductor': logging.getLogger('torch._inductor').level,
        }
        for key in self.prev_state.keys():
            setattr(torch_geometric.typing, key, False)
        for key in self.prev_log_level.keys():
            logging.getLogger(key).setLevel(self.log_level)

        warnings.filterwarnings('ignore', ".*the 'torch-scatter' package.*")

    def __exit__(self, *args, **kwargs):
        for key, value in self.prev_state.items():
            setattr(torch_geometric.typing, key, value)
        for key, value in self.prev_log_level.items():
            logging.getLogger(key).setLevel(value)
