import os
from pathlib import Path
from typing import Dict, Union

import torch

try:
    from huggingface_hub import (
        ModelCard,
        ModelCardData,
        ModelHubMixin,
        hf_hub_download,
    )
except ImportError:
    print('please install huggingface hub to use this mixin')

CONFIG_NAME = 'config.json'
MODEL_HUB_ORGANIZATION = "pytorch_geometric"
MODEL_WEIGHTS_NAME = 'model.pth'


class PyGModelHubMixin(ModelHubMixin):
    """
    Mixin for saving and loading models to Huggingface Model Hub

    Example of using this with Node2Vec and the Cora dataset from Planetoid:

    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import Node2Vec
    from torch_geometric.nn.model_hub import PyGModelHubMixin

    1. Define your class with the mixin:
    class N2V(Node2Vec, PyGModelHubMixin):
        def __init__(self,model_name, dataset_name, model_kwargs ):
            Node2Vec.__init__(self,**model_kwargs)
            PyGModelHubMixin.__init__(self, model_name,
                dataset_name, model_kwargs)
    2. instantiate your model:
    n2v = N2V(model_name='node2vec',
        dataset_name='Cora', model_kwargs=dict(
        edge_index=data.edge_index, embedding_dim=128,
        walk_length=20, context_size=10, walks_per_node=10,
        num_negative_samples=1, p=1, q=1, sparse=True))
    3. train model
    4. push to Huggingface:
    n2v.save_pretrained(local_file_path, push_to_hub=True,
        repo_id=[huggingface repo id])

    Load the model for inference:
    the required arguments are the repo id/local folder, and any model
    initialisation arguments that are not native python types (e.g
    Node2Vec requires the edge_index argument which is a tensor--
    this is not saved in model hub)

    model = N2V.from_pretrained([huggingface repo id/ local filepath],
        model_name='node2vec', dataset_name='Cora',
        edge_index=data.edge_index)
    """
    def __init__(self, model_name: str, dataset_name: str, model_kwargs: Dict):
        """

        Args:
            model_name (str): Name of the model as shown on the model card
            dataset_name (str): Name of dataset you trained model against
            model_kwargs (Dict): arguments that will be passed to the PyG model
        """
        ModelHubMixin.__init__(self)
        self.model_config = {
            k: v
            for k, v in model_kwargs.items() if type(v) in [str, int, float]
        }
        self.model_name = model_name
        self.dataset_name = dataset_name

    def construct_model_card(self, model_name: str,
                             dataset_name: str) -> ModelCard:
        card_data = ModelCardData(language='en', license='mit',
                                  library_name=MODEL_HUB_ORGANIZATION,
                                  tags=[MODEL_HUB_ORGANIZATION, model_name],
                                  datasets=dataset_name, model_name=model_name
                                  # metrics=['accuracy'],
                                  )
        card = ModelCard.from_template(
            card_data, model_description='info about model generate [TODO]')
        return card

    def _save_pretrained(self, save_directory: Union[Path, str]):
        """
        Args:
            save_directory (Path, str): local filepath to save model state dict
        """
        path = os.path.join(save_directory, MODEL_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)

    def save_pretrained(self, save_directory: str, push_to_hub: bool = False,
                        **kwargs):

        config = self.model_config
        if 'config' in kwargs.keys():
            kwargs.pop('config')

        ModelHubMixin.save_pretrained(self, save_directory, config,
                                      push_to_hub, **kwargs)
        model_card = self.construct_model_card(self.model_name,
                                               self.dataset_name)
        if push_to_hub:
            repo_id = kwargs.get('repo_id')
            model_card.push_to_hub(repo_id)

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        dataset_name="",
        model_name="",
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        """
        load trained model
        """
        map_location = torch.device(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, MODEL_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=MODEL_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )
        print(model_kwargs)
        if 'config' in model_kwargs.keys():
            config = model_kwargs['config']
            model_kwargs.pop('config')
            model_kwargs = {**model_kwargs, **config}

        print(model_kwargs)
        model = cls(dataset_name, model_name, model_kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model
