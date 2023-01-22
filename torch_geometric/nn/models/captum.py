from typing import Optional, Union

import torch

from torch_geometric.explain.algorithm.captum import \
    captum_output_to_dicts  # noqa
from torch_geometric.explain.algorithm.captum import to_captum_input  # noqa
from torch_geometric.explain.algorithm.captum import (
    CaptumHeteroModel,
    CaptumModel,
    MaskLevelType,
)
from torch_geometric.typing import Metadata


def to_captum_model(
    model: torch.nn.Module,
    mask_type: Union[str, MaskLevelType] = MaskLevelType.edge,
    output_idx: Optional[int] = None,
    metadata: Optional[Metadata] = None,
) -> Union[CaptumModel, CaptumHeteroModel]:
    r"""Converts a model to a model that can be used for
    `Captum.ai <https://captum.ai/>`_ attribution methods.

    Sample code for homogeneous graphs:

    .. code-block:: python

        from captum.attr import IntegratedGradients

        from torch_geometric.data import Data
        from torch_geometric.nn import GCN
        from torch_geometric.nn import to_captum_model, to_captum_input

        data = Data(x=(...), edge_index(...))
        model = GCN(...)
        ...  # Train the model.

        # Explain predictions for node `10`:
        mask_type="edge"
        output_idx = 10
        captum_model = to_captum_model(model, mask_type, output_idx)
        inputs, additional_forward_args = to_captum_input(data.x,
                                            data.edge_index,mask_type)

        ig = IntegratedGradients(captum_model)
        ig_attr = ig.attribute(inputs = inputs,
                               target=int(y[output_idx]),
                               additional_forward_args=additional_forward_args,
                               internal_batch_size=1)


    Sample code for heterogeneous graphs:

    .. code-block:: python

        from captum.attr import IntegratedGradients

        from torch_geometric.data import HeteroData
        from torch_geometric.nn import HeteroConv
        from torch_geometric.nn import (captum_output_to_dicts,
                                        to_captum_model, to_captum_input)

        data = HeteroData(...)
        model = HeteroConv(...)
        ...  # Train the model.

        # Explain predictions for node `10`:
        mask_type="edge"
        metadata = data.metadata
        output_idx = 10
        captum_model = to_captum_model(model, mask_type, output_idx, metadata)
        inputs, additional_forward_args = to_captum_input(data.x_dict,
                                            data.edge_index_dict, mask_type)

        ig = IntegratedGradients(captum_model)
        ig_attr = ig.attribute(inputs=inputs,
                               target=int(y[output_idx]),
                               additional_forward_args=additional_forward_args,
                               internal_batch_size=1)
        edge_attr_dict = captum_output_to_dicts(ig_attr, mask_type, metadata)


    .. note::
        For an example of using a Captum attribution method within PyG, see
        `examples/captum_explainability.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        captum_explainability.py>`_.

    Args:
        model (torch.nn.Module): The model to be explained.
        mask_type (str, optional): Denotes the type of mask to be created with
            a Captum explainer. Valid inputs are :obj:`"edge"`, :obj:`"node"`,
            and :obj:`"node_and_edge"`. (default: :obj:`"edge"`)
        output_idx (int, optional): Index of the output element (node or link
            index) to be explained. With :obj:`output_idx` set, the forward
            function will return the output of the model for the element at
            the index specified. (default: :obj:`None`)
        metadata (Metadata, optional): The metadata of the heterogeneous graph.
            Only required if explaning over a `HeteroData` object.
            (default: :obj: `None`)
    """
    if metadata is None:
        return CaptumModel(model, mask_type, output_idx)
    else:
        return CaptumHeteroModel(model, mask_type, output_idx, metadata)
