import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import requests
from huggingface_hub import (
    ModelCard,
    ModelCardData,
    PyTorchModelHubMixin,
    hf_hub_download,
)

CONFIG_NAME = 'config.json'
MODEL_HUB_ORGANIZATION = "pytorch_geometric"


def hfhub_model(PyGModel, dataset=None):
    # @wraps(PyGModel)
    class HFModel(PyGModel, PyTorchModelHubMixin):
        def __init__(self, **model_kwargs):
            PyGModel.__init__(self, **model_kwargs)
            PyTorchModelHubMixin.__init__(self)
            self.model_config = {
                k: v
                for k, v in model_kwargs.items()
                if type(v) in [str, int, float]
            }

        def construct_model_card(self):
            card_data = ModelCardData(
                language='en',
                license='mit',
                library_name=MODEL_HUB_ORGANIZATION,
                tags=[MODEL_HUB_ORGANIZATION, PyGModel.__name__],
                # todo this assumes a pyg dataset which has name attr
                datasets=dataset.name,
                model_name=PyGModel.__name__
                # metrics=['accuracy'],
            )
            card = ModelCard.from_template(
                card_data,
                model_description='info about model generate [TODO]')
            return card

        def save_pretrained(
            self,
            save_directory: str,
            # config: Optional[dict] = None,
            push_to_hub: bool = False,
            **kwargs,
        ):
            print(kwargs)
            config = self.model_config
            if 'config' in kwargs.keys():
                kwargs.pop('config')
            PyTorchModelHubMixin.save_pretrained(self, save_directory, config,
                                                 push_to_hub, **kwargs)

            model_card = self.construct_model_card()
            if push_to_hub:
                model_card.push_to_hub(kwargs.get('repo_id'))

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            force_download: bool = False,
            resume_download: bool = False,
            proxies: Optional[Dict] = None,
            use_auth_token: Optional[str] = None,
            cache_dir: Optional[str] = None,
            local_files_only: bool = False,
            **model_kwargs,
        ):
            model_id = pretrained_model_name_or_path

            revision = None
            if len(model_id.split("@")) == 2:
                model_id, revision = model_id.split("@")

            config_file: Optional[str] = None
            if os.path.isdir(model_id):
                if CONFIG_NAME in os.listdir(model_id):
                    config_file = os.path.join(model_id, CONFIG_NAME)
                else:
                    logging.warning(f"{CONFIG_NAME} not found "
                                    f"in {Path(model_id).resolve()}")
            else:
                try:
                    config_file = hf_hub_download(
                        repo_id=model_id,
                        filename=CONFIG_NAME,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        use_auth_token=use_auth_token,
                        local_files_only=local_files_only,
                    )
                except requests.exceptions.RequestException:
                    logging.warning(
                        f"{CONFIG_NAME} not found in HuggingFace Hub")

            if config_file is not None:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                model_kwargs.update(**config)

            return cls._from_pretrained(
                model_id,
                revision,
                cache_dir,
                force_download,
                proxies,
                resume_download,
                local_files_only,
                use_auth_token,
                **model_kwargs,
            )

    return HFModel


#
# class PyGModelHubMixin(PyTorchModelHubMixin):
#     def __init__(self, *args, **kwargs):
#         """
#         Mix this class with your torch-model class for ease process
#          of saving & loading from huggingface-hub.
#         Example:
#         ```python
#         >>> from huggingface_hub import PyTorchModelHubMixin
#         >>> class MyModel(nn.Module, PyTorchModelHubMixin):
#         ...     def __init__(self, **kwargs):
#         ...         super().__init__()
#         ...         self.config = kwargs.pop("config", None)
#         ...         self.layer = ...
#         ...     def forward(self, *args):
#         ...         return ...
#         >>> model = MyModel()
#         >>> model.save_pretrained(
#         ...     "mymodel", push_to_hub=False
#         >>> )  # Saving model weights in the directory
#         >>> model.push_to_hub(
#         ...     repo_id="mymodel", commit_message="model-1"
#         >>> )  # Pushing model-weights to hf-hub
#         >>> # Downloading weights from hf-hub & model will be
#              initialized from those weights
#         >>> model = MyModel.from_pretrained("username/mymodel@main")
#         ```
#         """
#
#     def _save_pretrained(self, save_directory):
#         """
#         Overwrite this method in case you don't want to save complete model,
#         rather some specific layers
#         """
#         path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
#         model_to_save = self.module if hasattr(self, "module") else self
#         torch.save(model_to_save.state_dict(), path)
#
#     @classmethod
#     def _from_pretrained(
#         cls,
#         model_id,
#         revision,
#         cache_dir,
#         force_download,
#         proxies,
#         resume_download,
#         local_files_only,
#         use_auth_token,
#         map_location="cpu",
#         strict=False,
#         **model_kwargs,
#     ):
#         """
#         Overwrite this method in case you wish to initialize your model in a
#         different way.
#         """
#         map_location = torch.device(map_location)
#
#         if os.path.isdir(model_id):
#             print("Loading weights from local directory")
#             model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
#         else:
#             model_file = hf_hub_download(
#                 repo_id=model_id,
#                 filename=PYTORCH_WEIGHTS_NAME,
#                 revision=revision,
#                 cache_dir=cache_dir,
#                 force_download=force_download,
#                 proxies=proxies,
#                 resume_download=resume_download,
#                 use_auth_token=use_auth_token,
#                 local_files_only=local_files_only,
#             )
#         model = cls(**model_kwargs)
#
#         state_dict = torch.load(model_file, map_location=map_location)
#         model.load_state_dict(state_dict, strict=strict)
#         model.eval()
#
#         return model
