from functools import wraps

from huggingface_hub import ModelCard, ModelCardData, PyTorchModelHubMixin

__MODEL_HUB_ORGANIZATION__ = "pytorch_geometric"


def hf_modelcard():
    card_data = ModelCardData(
        language='en',
        license='mit',
        library_name='timm',
        tags=['image-classification', 'resnet'],
        datasets='beans',
        metrics=['accuracy'],
    )
    card = ModelCard.from_template(
        card_data, model_description='This model does x + y...')

    return card


def hfhub_model(PyGModel, dataset):
    @wraps(PyGModel)
    class HFModel(PyGModel, PyTorchModelHubMixin):
        __MODEL_HUB_ORGANIZATION__ = "pytorch_geometric"

        def __init__(self, **model_kwargs):
            PyGModel.__init__(self, **model_kwargs)
            PyTorchModelHubMixin.__init__(self)

        def construct_model_card(self):
            card_data = ModelCardData(
                language='en',
                license='mit',
                library_name=__MODEL_HUB_ORGANIZATION__,
                tags=[__MODEL_HUB_ORGANIZATION__, PyGModel.__name__],
                datasets=dataset.
                name,  # todo this assumes a pyg dataset which has name attr
                # metrics=['accuracy'],
            )
            card = ModelCard.from_template(
                card_data, model_description='som edescription')
            return card

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
