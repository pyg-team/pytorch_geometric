import multiprocessing as mp
import warnings
from typing import Optional

import numpy as np
import torch
from torch import Tensor


def geodesic_distance(  # noqa: D417
    pos: Tensor,
    face: Tensor,
    src: Optional[Tensor] = None,
    dst: Optional[Tensor] = None,
    norm: bool = True,
    max_distance: Optional[float] = None,
    num_workers: int = 0,
    # Backward compatibility for `dest`:
    **kwargs: Optional[Tensor],
) -> Tensor:
    r"""Computes (normalized) geodesic distances of a mesh given by :obj:`pos`
    and :obj:`face`. If :obj:`src` and :obj:`dst` are given, this method only
    computes the geodesic distances for the respective source and target
    node-pairs.

    .. note::

        This function requires the :obj:`gdist` package.
        To install, run :obj:`pip install cython && pip install gdist`.

    Args:
        pos (torch.Tensor): The node positions.
        face (torch.Tensor): The face indices.
        src (torch.Tensor, optional): If given, only compute geodesic distances
            for the specified source indices. (default: :obj:`None`)
        dst (torch.Tensor, optional): If given, only compute geodesic distances
            for the specified target indices. (default: :obj:`None`)
        norm (bool, optional): Normalizes geodesic distances by
            :math:`\sqrt{\textrm{area}(\mathcal{M})}`. (default: :obj:`True`)
        max_distance (float, optional): If given, only yields results for
            geodesic distances less than :obj:`max_distance`. This will speed
            up runtime dramatically. (default: :obj:`None`)
        num_workers (int, optional): How many subprocesses to use for
            calculating geodesic distances.
            :obj:`0` means that computation takes place in the main process.
            :obj:`-1` means that the available amount of CPU cores is used.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Example:
        >>> pos = torch.tensor([[0.0, 0.0, 0.0],
        ...                     [2.0, 0.0, 0.0],
        ...                     [0.0, 2.0, 0.0],
        ...                     [2.0, 2.0, 0.0]])
        >>> face = torch.tensor([[0, 0],
        ...                      [1, 2],
        ...                      [3, 3]])
        >>> geodesic_distance(pos, face)
        [[0, 1, 1, 1.4142135623730951],
        [1, 0, 1.4142135623730951, 1],
        [1, 1.4142135623730951, 0, 1],
        [1.4142135623730951, 1, 1, 0]]
    """
    import gdist

    if 'dest' in kwargs:
        dst = kwargs['dest']
        warnings.warn("'dest' attribute in 'geodesic_distance' is deprecated "
                      "and will be removed in a future release. Use the 'dst' "
                      "argument instead.")

    max_distance = float('inf') if max_distance is None else max_distance

    if norm:
        area = (pos[face[1]] - pos[face[0]]).cross(
            pos[face[2]] - pos[face[0]],
            dim=1,
        )
        scale = float((area.norm(p=2, dim=1) / 2).sum().sqrt())
    else:
        scale = 1.0

    dtype = pos.dtype

    pos_np = pos.detach().cpu().to(torch.double).numpy()
    face_np = face.detach().t().cpu().to(torch.int).numpy()

    if src is None and dst is None:
        out = gdist.local_gdist_matrix(
            pos_np,
            face_np,
            max_distance * scale,
        ).toarray() / scale
        return torch.from_numpy(out).to(dtype)

    if src is None:
        src_np = torch.arange(pos.size(0), dtype=torch.int).numpy()
    else:
        src_np = src.detach().cpu().to(torch.int).numpy()

    dst_np = None if dst is None else dst.detach().cpu().to(torch.int).numpy()

    def _parallel_loop(
        pos_np: np.ndarray,
        face_np: np.ndarray,
        src_np: np.ndarray,
        dst_np: Optional[np.ndarray],
        max_distance: float,
        scale: float,
        i: int,
        dtype: torch.dtype,
    ) -> Tensor:
        s = src_np[i:i + 1]
        d = None if dst_np is None else dst_np[i:i + 1]
        out = gdist.compute_gdist(pos_np, face_np, s, d, max_distance * scale)
        out = out / scale
        return torch.from_numpy(out).to(dtype)

    num_workers = mp.cpu_count() if num_workers <= -1 else num_workers
    if num_workers > 0:
        with mp.Pool(num_workers) as pool:
            data = [(pos_np, face_np, src_np, dst_np, max_distance, scale, i,
                     dtype) for i in range(len(src_np))]
            outs = pool.starmap(_parallel_loop, data)
    else:
        outs = [
            _parallel_loop(pos_np, face_np, src_np, dst_np, max_distance,
                           scale, i, dtype) for i in range(len(src_np))
        ]

    out = torch.cat(outs, dim=0)

    if dst is None:
        out = out.view(-1, pos.size(0))

    return out
