import os
from unittest.mock import Mock

import pytest
import torch

from torch_geometric.testing import withPackage

pytest.importorskip("huggingface_hub")

from torch_geometric.nn.model_hub import PyGModelHubMixin  # noqa

REPO_NAME = "test"
MODEL_NAME = 'test_model'
DATASET_NAME = 'some_dataset'
CONFIG = {"hello": "world"}


@withPackage('huggingface_hub')
@pytest.fixture
def dummy_model_class():
    class DummyModel(torch.nn.Module, PyGModelHubMixin):
        def __init__(self, model_name, dataset_name, model_kwargs):
            torch.nn.Module.__init__(self)
            PyGModelHubMixin.__init__(self, model_name, dataset_name,
                                      model_kwargs)

        def load_state_dict(self, state_dict, strict):
            return None

    return DummyModel


@withPackage('huggingface_hub')
@pytest.fixture
def model(dummy_model_class):
    return dummy_model_class(MODEL_NAME, DATASET_NAME, CONFIG)


@withPackage('huggingface_hub')
def test_model_init(dummy_model_class):
    model = dummy_model_class(
        MODEL_NAME, DATASET_NAME, model_kwargs={
            **CONFIG, "tensor": torch.Tensor([1, 2, 3])
        })
    assert model.model_config == CONFIG


@withPackage('huggingface_hub')
def test_save_pretrained(model, tmp_path):
    save_directory = f"{str(tmp_path / REPO_NAME)}"
    # model._save_pretrained = Mock()
    model.save_pretrained(save_directory)
    files = os.listdir(save_directory)
    assert "model.pth" in files
    assert len(files) >= 1


@withPackage('huggingface_hub')
def test_save_pretrained_internal(model, tmp_path):
    save_directory = f"{str(tmp_path / REPO_NAME)}"
    model._save_pretrained = Mock()
    model.save_pretrained(save_directory)
    model._save_pretrained.assert_called_with(save_directory)


@withPackage('huggingface_hub')
def test_save_pretrained_with_push_to_hub(model, tmp_path):
    save_directory = f"{str(tmp_path / REPO_NAME)}"

    model.push_to_hub = Mock()
    model.construct_model_card = Mock()
    model._save_pretrained = Mock()  # disable _save_pretrained to speed-up

    # Not pushed to hub
    model.save_pretrained(save_directory)
    model.push_to_hub.assert_not_called()
    model.construct_model_card.assert_called_with(MODEL_NAME, DATASET_NAME)

    # Push to hub with repo_id
    model.save_pretrained(save_directory, push_to_hub=True, repo_id="CustomID",
                          config=CONFIG)
    model.push_to_hub.assert_called_with(repo_id="CustomID", config=CONFIG)

    # Push to hub with default repo_id (based on dir name)
    model.save_pretrained(save_directory, push_to_hub=True, config=CONFIG)
    model.push_to_hub.assert_called_with(repo_id=REPO_NAME, config=CONFIG)


@withPackage('huggingface_hub')
def test_from_pretrained(model, dummy_model_class, tmp_path):
    save_directory = f"{str(tmp_path / REPO_NAME)}"
    model.save_pretrained(save_directory)

    model = dummy_model_class.from_pretrained(save_directory)
    assert model.model_config == CONFIG


@withPackage('huggingface_hub')
def test_from_pretrained_internal(model, dummy_model_class, tmp_path,
                                  monkeypatch):
    hf_hub_download = Mock(side_effect='model')
    monkeypatch.setattr("torch_geometric.nn.model_hub.hf_hub_download",
                        hf_hub_download)
    monkeypatch.setattr("torch_geometric.nn.model_hub.torch.load",
                        lambda x, **kwargs: {'state_dict': 1})

    model = dummy_model_class._from_pretrained(
        model_id=MODEL_NAME, revision=None, cache_dir=None,
        force_download=False, proxies=None, resume_download=True,
        local_files_only=False, use_auth_token=False,
        dataset_name=DATASET_NAME, model_name=MODEL_NAME, map_location="cpu",
        strict=False, **CONFIG)

    assert hf_hub_download.call_count == 1
    assert model.model_config == CONFIG
