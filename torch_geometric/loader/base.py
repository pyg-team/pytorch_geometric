from typing import Any, Callable

from torch.utils.data.dataloader import _BaseDataLoaderIter


class DataLoaderIterator:
    r"""A data loader iterator extended by a simple post transformation
    function :meth:`transform_fn`. While the iterator may request items from
    different sub-processes, :meth:`transform_fn` will always be executed in
    the main process.

    This iterator is used in PyG's sampler classes, and is responsible for
    feature fetching and filtering data objects after sampling has taken place
    in a sub-process. This has the following advantages:

    * We do not need to share feature matrices across processes which may
      prevent any errors due to too many open file handles.
    * We can execute any expensive post-processing commands on the main thread
      with full parallelization power (which usually executes faster).
    * It lets us naturally support data already being present on the GPU.
    """
    def __init__(self, iterator: _BaseDataLoaderIter, transform_fn: Callable):
        self.iterator = iterator
        self.transform_fn = transform_fn

    def __iter__(self) -> 'DataLoaderIterator':
        return self

    def _reset(self, loader: Any, first_iter: bool = False):
        self.iterator._reset(loader, first_iter)

    def __len__(self) -> int:
        return len(self.iterator)

    def __next__(self) -> Any:
        return self.transform_fn(next(self.iterator))
