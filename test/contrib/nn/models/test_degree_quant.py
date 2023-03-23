import pytest
from torch import nn

# from torch_geometric.contrib.degree_quant.data_handler import get_dataset
from torch_geometric.contrib.degree_quant.models import GAT, GCN, GIN
# from torch_geometric.contrib.degree_quant.training import (
# cross_validation_with_val_set
# )
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
from torch_geometric.datasets import KarateClub


@pytest.mark.parametrize('qypte', ['INT4', 'INT8'])
@pytest.mark.parametrize('dataset_name', ['MUTAG', 'IMDB-BINARY'])
def test_layers(qypte, dataset_name):

    argsINT8 = {
        "qypte": "INT8",
        "ste": False,
        "momentum": False,
        "percentile": 0.001,
        "dq": True,
        "sample_prop": None,
    }

    # modelparams = {
    #     "folds": 3,
    #     "epochs": 1,
    #     "batch_size": 16,
    #     "lr": 0.0005,
    #     "lr_decay_factor": 0.5,
    #     "lr_decay_step_size": 50,
    #     "weight_decay": 0.0002,
    #     "writer": None,
    #     "logger": None
    # }

    dataset = KarateClub()

    assert dataset.num_classes == 4
    assert dataset.num_features == 34

    GIN_model = GIN(dataset.num_features, dataset.num_classes, num_layers=2,
                    hidden=64, **argsINT8)
    GCN_model = GCN(dataset.num_features, dataset.num_classes, num_layers=2,
                    hidden=64, **argsINT8)
    GAT_model = GAT(dataset.num_features, dataset.num_classes, num_layers=2,
                    hidden=64, **argsINT8)

    assert isinstance(GIN_model, nn.Module) is True
    assert isinstance(GCN_model, nn.Module) is True
    assert isinstance(GAT_model, nn.Module) is True
