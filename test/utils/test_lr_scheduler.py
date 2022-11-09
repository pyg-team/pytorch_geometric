import pytest
import torch
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, ReduceLROnPlateau

from torch_geometric.utils import lr_scheduler_resolver


@pytest.mark.parametrize('scheduler_args', [
    ("constant_with_warmup", LambdaLR),
    ("linear_with_warmup", LambdaLR),
    ("cosine_with_warmup", LambdaLR),
    ("cosine_with_warmup_restarts", LambdaLR),
    ("polynomial_with_warmup", LambdaLR),
    ("constant", ConstantLR),
    ('ReduceLROnPlateau', ReduceLROnPlateau),
])
def test_lr_scheduler_resolver(scheduler_args):
    scheduler_name, scheduler_cls = scheduler_args
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lr_scheduler = lr_scheduler_resolver(
        scheduler_name,
        optimizer,
        num_training_steps=100,
    )
    assert isinstance(lr_scheduler, scheduler_cls)
