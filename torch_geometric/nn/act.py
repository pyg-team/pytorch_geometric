import torch

ACTS = {
    'relu': torch.nn.ReLU(inplace=True),
    'elu': torch.nn.ELU(inplace=True),
    'leaky_relu': torch.nn.LeakyReLU(inplace=True),
}
