from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.nn.models.gpse import GPSE
from torch_geometric.transforms import BaseTransform, VirtualNode


@functional_transform('add_gpse')
class AddGPSE(BaseTransform):
    r"""Adds the GPSE encoding from the `"Graph Positional and Structural
    Encoder" <https://arxiv.org/abs/2307.07107>`_ paper to the given graph
    (functional name: :obj:`add_gpse`).
    To be used with a :class:`~torch_geometric.nn.GPSE` model, which generates
    the actual encodings.

    Args:
        model (GPSE): The pre-trained GPSE model.
        use_vn (bool, optional): Whether to use virtual nodes.
            (default: :obj:`True`)
        rand_type (str, optional): Type of random features to use. Options are
            :obj:`NormalSE`, :obj:`UniformSE`, :obj:`BernoulliSE`.
            (default: :obj:`NormalSE`)

    """
    def __init__(self, model: GPSE, use_vn: bool = True,
                 rand_type: str = 'NormalSE'):
        self.model = model
        self.use_vn = use_vn
        self.vn = VirtualNode()
        self.rand_type = rand_type

    def __call__(self, data: Data) -> Data:
        from torch_geometric.nn.models.gpse import gpse_process

        data_vn = self.vn(data.clone()) if self.use_vn else data.clone()
        batch_out = gpse_process(self.model, data_vn, 'NormalSE', self.use_vn)
        batch_out = batch_out.to('cpu', non_blocking=True)
        data.pestat_GPSE = batch_out[:-1] if self.use_vn else batch_out

        return data
