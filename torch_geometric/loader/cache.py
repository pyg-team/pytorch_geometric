from typing import Any, Callable, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader


class CachedLoader:
    r"""A caching loader intended to use in layer-wise inference
    performed on GPU. During the first walk through the data stored in
    :class:`torch.utils.data.DataLoader`, selected attributes from batches
    are transferred to GPU memory and cached. Default attributes are
    `n_id`, `edge_index` and `batch_size`. User can register custom
    attribute hooks, but only before iterating over loader.

    Args:
        loader (torch.utils.data.DataLoader): The wrapped data loader.
        device (torch.device): The device to load the data to.
    """
    def __init__(self, loader: DataLoader, device: torch.device):
        self.n_batches = len(loader)
        self.iter = iter(loader)
        self.cache_filled = False
        self.hook_reg_open = True
        self.device = device
        self.attr_hooks = []
        self.cache = []

        # register default hooks
        self.register_attr_hooks([
            lambda b: b.n_id,
            lambda b: b.adj_t if hasattr(b, 'adj_t') else b.edge_index,
            lambda b: b.batch_size])

    def _check_if_reg_is_open(self) -> None:
        if not self.hook_reg_open:
            raise RuntimeError('Cannot register nor reset attributes hooks '
                               'after starting iteration over `CachedLoader`')

    def reset_attr_hooks(self) -> None:
        r"""Resets attribute hooks if possible.

        Raises:
            RuntimeError: If hook registration is closed.
        """
        self._check_if_reg_is_open()
        self.attr_hooks = []

    def register_attr_hook(self, hook: Callable[[Any], Any]) -> None:
        r"""Registers attribute hook.

        Args:
            hook (Callable[[Any], Any]): The attribute hook to register.

        Raises:
            RuntimeError: If hook registration is closed.

        Example:

        .. code-block:: python

            cached_loader = CachedLoader(wrapped_loader, target_device)
            cached_loader.register_attr_hook(lambda b: b.n_id)
        """
        self._check_if_reg_is_open()
        self.attr_hooks.append(hook)

    def register_attr_hooks(self, hooks: List[Callable[[Any], Any]]) -> None:
        r"""Registers attribute hooks, overwriting currently existing hooks.

        Args:
            hooks (List[Callable[[Any], Any]]): The attribute hooks to
                register.

        Raises:
            RuntimeError: If hook registration is closed.
        """
        self.reset_attr_hooks()
        for hook in hooks:
            self.register_attr_hook(hook)

    def _cache_batch_attrs(self, batch: Any) -> Any:
        attrs = []
        for hook in self.attr_hooks:
            attr = hook(batch)
            if isinstance(attr, Tensor):
                attr = attr.to(self.device)
            attrs.append(attr)
        self.cache.append(attrs)
        return attrs

    def __iter__(self) -> Any:
        self.hook_reg_open = False
        return self

    def __next__(self) -> List[Any]:
        try:
            batch = next(self.iter)
            if not self.cache_filled:
                batch = self._cache_batch_attrs(batch)
            return batch
        except StopIteration as exc:
            self.cache_filled = True
            self.iter = iter(self.cache)
            raise StopIteration from exc

    def __len__(self) -> int:
        return self.n_batches

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'
