import torch

graph_type_full = ['invest', 'charge', 'supply', 'audit']
epochs = 200

graph_type_etypes = {
    'invest': [  # src->tgt
        ('L', 'L-U', 'U'),
        ('U', 'U-L', 'L'),
        ('L', 'L-P', 'P'),
        ('P', 'P-L', 'L'),
        ('L', 'L-R', 'R'),
        ('R', 'R-L', 'L'),
        ('L', 'L-L', 'L'),
    ],
    'charge': [
        ('L', 'L-U', 'U'),
        ('U', 'U-L', 'L'),
        ('L', 'L-P', 'P'),
        ('P', 'P-L', 'L'),
        ('L', 'L-R', 'R'),
        ('R', 'R-L', 'L'),
        ('L', 'L-L', 'L'),
    ],
    'supply': [
        ('L', 'L-U', 'U'),
        ('U', 'U-L', 'L'),
        ('L', 'L-L', 'L'),
    ],
    'audit': [
        ('L', 'L-A', 'A'),
        ('A', 'A-L', 'L'),
    ]
}

split_dicts = {0: 2018, 1: 2019, 2: 2020, 3: 2021, 4: 2022}
device = torch.device("cpu")
feature_path = "feature/feature.csv"

times = 1
