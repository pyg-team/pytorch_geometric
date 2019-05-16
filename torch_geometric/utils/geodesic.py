import torch
import numpy as np

try:
    import gdist
except ImportError:
    gdist = None


def geodesic_distance(pos,
                      face,
                      src=None,
                      dest=None,
                      norm=True,
                      max_distance=None):
    r"""Computes (normalized) geodesic distances of a mesh given by :obj:`pos`
    and :obj:`face`. If :obj:`src` and :obj:`dest` are given, this method only
    computes the geodesic distances for the respective source and target
    node-pairs.

    .. note::

        This function requires the :obj:`gdist` package.
        To install, run :obj:`pip install cython && pip install gdist`.

    Args:
        pos (Tensor): The node positions.
        face (LongTensor): The face indices.
        src (LongTensor, optional): If given, only compute geodesic distances
            for the specified source indices. (default: :obj:`None`)
        dest (LongTensor, optional): If given, only compute geodesic distances
            for the specified target indices. (default: :obj:`None`)
        norm (bool, optional): Normalizes geodesic distances by
            :math:`\sqrt{\textrm{area}(\mathcal{M})}`. (default: :obj:`True`)
        max_distance (float, optional): If given, only yields results for
            geodesic distances less than :obj:`max_distance`. This will speed
            up runtime dramatically. (default: :obj:`None`)

    :rtype: Tensor
    """

    if gdist is None:
        raise ImportError('Package "gdist" could not be found.')

    max_distance = float('inf') if max_distance is None else max_distance

    if norm:
        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        norm = (area.norm(p=2, dim=1) / 2).sum().sqrt().item()
    else:  # pragma: no cover
        norm = 1.0

    if src is None:
        src = np.arange(pos.size(0), dtype=np.int32)
    else:
        src = src.detach().cpu().to(torch.int).numpy()

    dest = None if dest is None else dest.detach().cpu().to(torch.int).numpy()

    dtype = pos.dtype
    pos = pos.detach().cpu().to(torch.double).numpy()
    face = face.detach().t().cpu().to(torch.int).numpy()

    outs = []
    for i in range(len(src)):
        s = src[i:i + 1]
        d = None if dest is None else dest[i:i + 1]

        out = gdist.compute_gdist(pos, face, s, d, max_distance * norm) / norm
        out = torch.from_numpy(out).to(dtype)
        outs.append(out)

    out = torch.cat(outs, dim=0)

    if dest is None:
        out = out.view(-1, pos.shape[0])

    return out
