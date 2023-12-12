from abc import abstractmethod
from typing import Optional, Union, Dict
import torch
from torch import Tensor
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain import GenerativeExplanation

class XGNNExplainer(ExplainerAlgorithm):
    r"""The XGNN-Explainer model from the `"XGNN: Towards Model-Level Explanations of Graph Neural Networks"
    <https://arxiv.org/abs/2006.02587>`_ paper for training a graph generator so that
    the generated graph patterns maximize a certain prediction of the model.

    XGNN-Explainer interprets GNN models by training a graph generator, which
    creates graph patterns that maximize the predictive outcome for a specific
    class. This approach provides a model-level explanation, offering insights
    into what input patterns lead to certain predictions by GNNs.

    .. note::

        For an example of using :class:`XGNNExplainer`, see
        `examples/explain/xgnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/xgnn_explainer.py>`_.

    The explainer trains a generative model, iterating over steps to progressively
    build a graph that maximizes the class score of the target class. The generative
    model can be customized and is a key component of the explanation process.

    Args:
        epochs (int, optional): The number of epochs for training the generative model.
            (default: :obj:`100`)
        lr (float, optional): The learning rate for training the generative model.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to configure the training
            process of the generative model.
    """
    
    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.config = kwargs
        
    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[str, Tensor]],
        edge_index: Union[Tensor, Dict[str, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> GenerativeExplanation:
        r"""Computes the generative explanation for each class.
        
        Args:
            model (torch.nn.Module): The model to explain.
            x (Union[Tensor, Dict[str, Tensor]]): The input node features.
            edge_index (Union[Tensor, Dict[str, Tensor]]): The edge indices.
            target (Tensor): The target tensor for the explanation.
            index (Optional[Union[int, Tensor]], optional): The index of the node or graph to explain,
            the index does not matter for this explainer, and is only here for the sake of integration with
            other classes.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[Explanation, HeteroExplanation]: The explanation result.
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")
            
        if index is not None:
            raise ValueError(f"Index not supported in '{self.__class__.__name__}'")
        
        generative_models = dict()
        for t in torch.unique(target):
            generative_models[t] = self.train_generative_model(model, 
                                                               for_class = t, 
                                                               num_epochs = self.epochs, 
                                                               learning_rate = self.lr, 
                                                               **kwargs)
        return GenerativeExplanation(model = model, 
                                     generative_models = generative_models)

    @abstractmethod
    def train_generative_model(self, model, for_class):
        r""" Abstract method to train the generative model. Must be implemented in subclasses.
        
        Args:
            model: The model to explain.
            for_class: The class for which the explanation is generated.
        """
        pass
    
    def supports(self) -> bool:
        return True