import torch

# 加载图数据和特征数据
split_dicts = {
    0: {
        'train': 2018,
        'valid': 2018,
        'test': 2018
    },
    1: {
        'train': 2019,
        'valid': 2019,
        'test': 2019
    },
    2: {
        'train': 2020,
        'valid': 2020,
        'test': 2020
    },
    3: {
        'train': 2021,
        'valid': 2021,
        'test': 2021
    },
    4: {
        'train': 2022,
        'valid': 2022,
        'test': 2022
    }
}

# 设置参数
num_heads = [8]
dropout = 0.5
epochs = 200
run_times = 5
num_classes = 2
hidden_size = 8
edge_type = ['L-L-L', 'L-U-L', 'L-P-L', 'L-R-L', 'L-A-L']
feature_shape = 169

# 设置CPU或GPU
device = torch.device("cpu")
