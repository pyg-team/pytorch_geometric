from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch_geometric.explain import ExplainerConfig, GenerativeExplanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
    
class XGNNExplainer(ExplainerAlgorithm):
    r"""The XGNN-Explainer model from the `"XGNN: Towards Model-Level Explanations of Graph Neural Networks"
    <https://arxiv.org/abs/2006.02587>`_ paper for training a graph generator so that
    the generated graph patterns maximize a certain prediction of the model.

    .. note::

        For an example of using :class:`XGNNExplainer`, see
        `examples/explain/xgnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/xgnn_explainer.py>`_.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """
    
    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> GenerativeExplanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")
        # for each class get a model

        print("debug: xgnn forward (file: xgnn_explainer.py)")
        
        generative_models = dict()
        for t in torch.unique(target):
            generative_models[t] = self.train_generative_model(model, 
                                                               for_class = t, 
                                                               num_epochs = 100, 
                                                               learning_rate = 0.01, 
                                                               **kwargs)
        # self._clean_model(model)
        return GenerativeExplanation(model = model, generative_models = generative_models)

    def train_generative_model(self, model, for_class, num_epochs = 100, learning_rate = 0.01):
        """
        Trains the generative model.

        Args:
            model: predicitve model used to train the generative model
            for_class: the class to train the generative model for
            epochs (int, optional): The number of epochs to train.
                (default: :obj:`100`)
            learning_rate: lr (float, optional): The learning rate to apply.
                (default: :obj:`0.01`)

        Returns:
            None
        """

        # TODO: throw errors, this is a base class
        print("debug: xgnn train")
        pass


    def supports(self) -> bool:
        return True
    
    