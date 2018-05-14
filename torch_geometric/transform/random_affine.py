import torch


class RandomAffine(object):
    r"""Random affine transformation of all node positions.

    Args:
        degrees (sequence or float or int, optional): M degrees to
            select from. If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not rotate by default. (default:
            :obj:`None`)
        translate (sequence or float or int, optional): Sequence of maximum
            absolute translations for each dimension. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default


