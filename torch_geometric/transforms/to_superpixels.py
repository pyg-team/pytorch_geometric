import torch
from torch_scatter import scatter_mean
from torch_geometric.data import Data

try:
    from skimage.segmentation import slic
except ImportError:
    slic = None


class ToSLIC(object):
    r"""Converts an image to a superpixel representation using the
    :meth:`skimage.segmentation.slic` algorithm, resulting in a
    :obj:`torch_geometric.data.Data` object holding the centroids of
    superpixels in :obj:`pos` and their mean color in :obj:`x`.

    This transform can be used with any :obj:`torchvision` dataset.

    Example::

        from torchvision.datasets import MNIST
        import torchvision.transforms as T
        from torch_geometric.transforms import ToSLIC

        transform = T.Compose([T.ToTensor(), ToSLIC(n_segments=75)])
        dataset = MNIST('/tmp/MNIST', download=True, transform=transform)

    Args:
        add_seg (bool, optional): If set to `True`, will add the segmentation
            result to the data object. (default: :obj:`False`)
        add_img (bool, optional): If set to `True`, will add the input image
            to the data object. (default: :obj:`False`)
        **kwargs (optional): Arguments to adjust the output of the SLIC
            algorithm. See the `SLIC documentation
            <https://scikit-image.org/docs/dev/api/skimage.segmentation.html
            #skimage.segmentation.slic>`_ for an overview.
    """
    def __init__(self, add_seg=False, add_img=False, **kwargs):

        if slic is None:
            raise ImportError('`ToSlic` requires `scikit-image`.')

        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def __call__(self, img):
        img = img.permute(1, 2, 0)
        h, w, c = img.size()

        seg = slic(img.to(torch.double).numpy(), start_label=0, **self.kwargs)
        seg = torch.from_numpy(seg)

        x = scatter_mean(img.view(h * w, c), seg.view(h * w), dim=0)

        pos_y = torch.arange(h, dtype=torch.float)
        pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
        pos_x = torch.arange(w, dtype=torch.float)
        pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)

        pos = torch.stack([pos_x, pos_y], dim=-1)
        pos = scatter_mean(pos, seg.view(h * w), dim=0)

        data = Data(x=x, pos=pos)

        if self.add_seg:
            data.seg = seg.view(1, h, w)

        if self.add_img:
            data.img = img.permute(2, 0, 1).view(1, c, h, w)

        return data
