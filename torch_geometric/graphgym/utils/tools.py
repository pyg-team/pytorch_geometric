import random
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class dummy_context():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
