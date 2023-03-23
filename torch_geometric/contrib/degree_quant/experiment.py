from .data_handler import get_dataset
from .models import GAT, GCN, GIN
from .training import cross_validation_with_val_set

# from gat_conv import GATConvMultiQuant from gcn_conv import GCNConvMultiQuant
# from gin_conv import GINConvMultiQuant

# Simple Implementation of GIN, GCN and GAT on the MUTAG dataset We convert the
# FP32 to INT8 in this experiment

argsINT8 = {
    "qypte": "INT8",
    "ste": False,
    "momentum": False,
    "percentile": 0.001,
    "dq": True,
    "sample_prop": None,
}

modelparams = {
    "folds": 3,
    "epochs": 1,
    "batch_size": 16,
    "lr": 0.0005,
    "lr_decay_factor": 0.5,
    "lr_decay_step_size": 50,
    "weight_decay": 0.0002,
    "writer": None,
    "logger": None
}

dataset = get_dataset("./redditbinary/", 'MUTAG', sparse=True)

print(dataset)

GIN_model = GIN(dataset.num_features, dataset.num_classes, num_layers=2,
                hidden=64, **argsINT8)
GCN_model = GCN(dataset.num_features, dataset.num_classes, num_layers=2,
                hidden=64, **argsINT8)
GAT_model = GAT(dataset.num_features, dataset.num_classes, num_layers=2,
                hidden=64, **argsINT8)

GIN_metrics = cross_validation_with_val_set(dataset, GIN_model, **modelparams)

GCN_metrics = cross_validation_with_val_set(dataset, GCN_model, **modelparams)

GAT_metrics = cross_validation_with_val_set(dataset, GAT_model, **modelparams)

# model = GCNConvMultiQuant(i
