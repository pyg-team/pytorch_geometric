import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch

try:
    from huggingface_hub import (
        ModelCard,
        ModelCardData,
        ModelHubMixin,
        hf_hub_download,
    )
except ImportError:
    raise ImportError("'PyGModelHubMixin' requires 'huggingface_hub'")

CONFIG_NAME = 'config.json'
MODEL_HUB_ORGANIZATION = "pytorch_geometric"
MODEL_WEIGHTS_NAME = 'model.pth'


class PyGModelHubMixin(ModelHubMixin):
    r"""
    Mixin for saving and loading models to Huggingface Model Hub

    Args:
        model_name (str): Name of the model as shown on the model card
        dataset_name (str): Name of the dataset you trained model against
        model_kwargs (Dict): Arguments that will be passed to the PyG model

    Example of using this with Node2Vec and the Cora dataset from Planetoid:

    .. code-block:: python

      from torch_geometric.datasets import Planetoid
      from torch_geometric.nn import Node2Vec
      from torch_geometric.nn.model_hub import PyGModelHubMixin

      # Define your class with the mixin:
      class N2V(Node2Vec, PyGModelHubMixin):
          def __init__(self,model_name, dataset_name, model_kwargs ):
              Node2Vec.__init__(self,**model_kwargs)
              PyGModelHubMixin.__init__(self, model_name,
                  dataset_name, model_kwargs)

      # instantiate your model:
      n2v = N2V(model_name='node2vec',
          dataset_name='Cora', model_kwargs=dict(
          edge_index=data.edge_index, embedding_dim=128,
          walk_length=20, context_size=10, walks_per_node=10,
          num_negative_samples=1, p=1, q=1, sparse=True))

      # train model
      ...

      # push to Huggingface:
      repo_id = ... # your repo id
      n2v.save_pretrained(local_file_path, push_to_hub=True,
          repo_id=repo_id)

      # Load the model for inference:
      # The required arguments are the repo id/local folder, and any model
      # initialisation arguments that are not native python types (e.g
      # Node2Vec requires the edge_index argument which is a tensor--
      # this is not saved in model hub)

      model = N2V.from_pretrained( repo_id,
          model_name='node2vec', dataset_name='Cora',
          edge_index=data.edge_index)

    ..note::
        At the moment the model card is fairly basic--
        override the `construct_model_card` method if you want to have
        a more detailed model card

    """
    def __init__(self, model_name: str, dataset_name: str, model_kwargs: Dict):
        ModelHubMixin.__init__(self)
        # Huggingface Hub api only accepts saving the config as a dict.
        # If the model is instantiated with non-native python types
        # such as torch Tensors (node2vec being an example), we have to remove
        # these as they are not json serialisable
        self.model_config = {
            k: v
            for k, v in model_kwargs.items() if type(v) in [str, int, float]
        }
        self.model_name = model_name
        self.dataset_name = dataset_name

    def construct_model_card(self, model_name: str,
                             dataset_name: str) -> ModelCard:
        card_data = ModelCardData(
            language='en',
            license='mit',
            library_name=MODEL_HUB_ORGANIZATION,
            tags=[MODEL_HUB_ORGANIZATION, model_name],
            datasets=dataset_name,
            model_name=model_name,
        )
        card = ModelCard.from_template(card_data)
        return card

    def _save_pretrained(self, save_directory: Union[Path, str]):
        r"""
        Args:
            save_directory (Path or str): local filepath to
                save model state dict
        """
        path = os.path.join(save_directory, MODEL_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)

    def save_pretrained(self, save_directory: Union[str, Path],
                        push_to_hub: bool = False, repo_id=None, **kwargs):
        r"""
        Save a trained model to a local directory or to huggingface model hub
        Args:
            save_directory (`str` or `Path`):
                File directory in which you want to save weights.
            push_to_hub: Whether to save weights on model hub
            repo_id: the repository you want to push to
                (will default to the name of `save_directory` in your
                namespace).
        """
        config = self.model_config
        # due to way huggingface hub handles the loading/saving of models,
        # the model config can end up in one of the items in the kwargs
        # this has to be removed to prevent a duplication of arguments to
        # ModelHubMixin.save_pretrained
        kwargs.pop('config', None)

        ModelHubMixin.save_pretrained(self, save_directory, config,
                                      push_to_hub, repo_id=repo_id, **kwargs)
        model_card = self.construct_model_card(self.model_name,
                                               self.dataset_name)
        if push_to_hub:
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

        config = model_kwargs.pop('config', None)
        if config is not None:
            model_kwargs = {**model_kwargs, **config}

        model = cls(dataset_name, model_name, model_kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ):
        r"""
        Download and instantiate a model from the Hugging Face Hub.

        Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
        Can be either:
            - A string, the `model id` of a pretrained model
              hosted inside a model repo on huggingface.co.
              Valid model ids can be located at the root-level,
              like `bert-base-uncased`, or namespaced under a
              user or organization name, like
              `dbmdz/bert-base-german-cased`.
            - You can add `revision` by appending `@` at the end
              of model_id simply like this:
              `dbmdz/bert-base-german-cased@main` Revision is
              the specific model version to use. It can be a
              branch name, a tag name, or a commit id, since we
              use a git-based system for storing models and
              other artifacts on huggingface.co, so `revision`
              can be any identifier allowed by git.
            - A path to a `directory` containing model weights
              saved using
              [`~transformers.PreTrainedModel.save_pretrained`],
              e.g., `./my_model_directory/`.
            - `None` if you are both providing the configuration
              and state dictionary (resp. with keyword arguments
              `config` and `state_dict`).
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the (re-)download of the model weights
            and configuration files, overriding the cached versions
            if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether to delete incompletely received files. Will
            attempt to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or
            endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are
            used on each request.
        token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote
            files. If `True`, will use the token generated when
            running `transformers-cli login` (stored in
            `~/.huggingface`). It is **required** if you
            want to use a private model
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory in which a downloaded pretrained
            model configuration should be cached if the standard
            cache should not be used.
        local_files_only(`bool`, *optional*, defaults to `False`):
            Whether to only look at local files (i.e., do not try to
            download the model).
        model_kwargs (`Dict`, *optional*):
            model_kwargs will be passed to the model during
            initialization
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            force_download,
            resume_download,
            proxies,
            token,
            cache_dir,
            local_files_only,
            **model_kwargs,
        )
