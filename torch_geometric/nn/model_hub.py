import os.path as osp
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

try:
    from huggingface_hub import ModelHubMixin, hf_hub_download
except ImportError:
    ModelHubMixin = object
    hf_hub_download = None

CONFIG_NAME = 'config.json'
MODEL_HUB_ORGANIZATION = "pytorch_geometric"
MODEL_WEIGHTS_NAME = 'model.pth'
TAGS = ['graph-machine-learning']


class PyGModelHubMixin(ModelHubMixin):
    r"""A mixin for saving and loading models to the
    `Huggingface Model Hub <https://huggingface.co/docs/hub/index>`_.

    .. code-block:: python

       from torch_geometric.datasets import Planetoid
       from torch_geometric.nn import Node2Vec
       from torch_geometric.nn.model_hub import PyGModelHubMixin

       # Define your class with the mixin:
       class N2V(Node2Vec, PyGModelHubMixin):
           def __init__(self,model_name, dataset_name, model_kwargs):
               Node2Vec.__init__(self,**model_kwargs)
               PyGModelHubMixin.__init__(self, model_name,
                   dataset_name, model_kwargs)

       # Instantiate your model:
       n2v = N2V(model_name='node2vec',
           dataset_name='Cora', model_kwargs=dict(
           edge_index=data.edge_index, embedding_dim=128,
           walk_length=20, context_size=10, walks_per_node=10,
           num_negative_samples=1, p=1, q=1, sparse=True))

       # Train the model:
       ...

       # Push to the HuggingFace hub:
       repo_id = ...  # your repo id
       n2v.save_pretrained(
           local_file_path,
           push_to_hub=True,
           repo_id=repo_id,
        )

       # Load the model for inference:
       # The required arguments are the repo id/local folder, and any model
       # initialisation arguments that are not native python types (e.g
       # Node2Vec requires the edge_index argument which is not stored in the
       # model hub).
       model = N2V.from_pretrained(
           repo_id,
           model_name='node2vec',
           dataset_name='Cora',
           edge_index=data.edge_index,
       )

    Args:
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset the model was trained against.
        model_kwargs (Dict[str, Any]): The arguments to initialise the model.
    """
    def __init__(self, model_name: str, dataset_name: str, model_kwargs: Dict):
        ModelHubMixin.__init__(self)

        # Huggingface Hub API only accepts saving the config as a dict.
        # If the model is instantiated with non-native python types
        # such as torch Tensors (node2vec being an example), we have to remove
        # these as they are not json serialisable
        self.model_config = {
            k: v
            for k, v in model_kwargs.items() if type(v) in [str, int, float]
        }
        self.model_name = model_name
        self.dataset_name = dataset_name

    def construct_model_card(self, model_name: str, dataset_name: str) -> Any:
        from huggingface_hub import ModelCard, ModelCardData
        card_data = ModelCardData(
            language='en',
            license='mit',
            library_name=MODEL_HUB_ORGANIZATION,
            tags=TAGS,
            datasets=dataset_name,
            model_name=model_name,
        )
        card = ModelCard.from_template(card_data)
        return card

    def _save_pretrained(self, save_directory: Union[Path, str]):
        path = osp.join(save_directory, MODEL_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), path)

    def save_pretrained(self, save_directory: Union[str, Path],
                        push_to_hub: bool = False,
                        repo_id: Optional[str] = None, **kwargs):
        r"""Save a trained model to a local directory or to the HuggingFace
        model hub.

        Args:
            save_directory (str): The directory where weights are saved.
            push_to_hub (bool, optional): If :obj:`True`, push the model to the
                HuggingFace model hub. (default: :obj:`False`)
            repo_id (str, optional): The repository name in the hub.
                If not provided will default to the name of
                :obj:`save_directory` in your namespace. (default: :obj:`None`)
            **kwargs: Additional keyword arguments passed to
                :meth:`huggingface_hub.ModelHubMixin.save_pretrained`.
        """
        config = self.model_config
        # due to way huggingface hub handles the loading/saving of models,
        # the model config can end up in one of the items in the kwargs
        # this has to be removed to prevent a duplication of arguments to
        # ModelHubMixin.save_pretrained
        kwargs.pop('config', None)

        super().save_pretrained(
            save_directory=save_directory,
            config=config,
            push_to_hub=push_to_hub,
            repo_id=repo_id,
            **kwargs,
        )
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
        token,
        dataset_name='',
        model_name='',
        map_location='cpu',
        strict=False,
        **model_kwargs,
    ):
        map_location = torch.device(map_location)

        if osp.isdir(model_id):
            model_file = osp.join(model_id, MODEL_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=MODEL_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
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
    ) -> Any:
        r"""Downloads and instantiates a model from the HuggingFace hub.

        Args:
            pretrained_model_name_or_path (str): Can be either:

                - The :obj:`model_id` of a pretrained model hosted inside the
                  HuggingFace hub.

                - You can add a :obj:`revision` by appending :obj:`@` at the
                  end of :obj:`model_id` to load a specific model version.

                - A path to a directory containing the saved model weights.

                - :obj:`None` if you are both providing the configuration
                  :obj:`config` and state dictionary :obj:`state_dict`.

            force_download (bool, optional): Whether to force the
                (re-)download of the model weights and configuration files,
                overriding the cached versions if they exist.
                (default: :obj:`False`)
            resume_download (bool, optional): Whether to delete incompletely
                received files. Will attempt to resume the download if such a
                file exists. (default: :obj:`False`)
            proxies (Dict[str, str], optional): A dictionary of proxy servers
                to use by protocol or endpoint, *e.g.*,
                :obj:`{'http': 'foo.bar:3128', 'http://host': 'foo.bar:4012'}`.
                The proxies are used on each request. (default: :obj:`None`)
            token (str or bool, optional): The token to use as HTTP bearer
                authorization for remote files. If set to :obj:`True`, will use
                the token generated when running :obj:`transformers-cli login`
                (stored in :obj:`~/.huggingface`). It is **required** if you
                want to use a private model. (default: :obj:`None`)
            cache_dir (str, optional): The path to a directory in which a
                downloaded model configuration should be cached if the
                standard cache should not be used. (default: :obj:`None`)
            local_files_only (bool, optional): Whether to only look at local
                files, *i.e.* do not try to download the model.
                (default: :obj:`False`)
            **model_kwargs: Additional keyword arguments passed to the
                model during initialization.
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        )
