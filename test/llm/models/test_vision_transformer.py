import torch

from torch_geometric.llm.models import VisionTransformer
from torch_geometric.testing import onlyFullTest, withCUDA, withPackage


@withCUDA
@onlyFullTest
@withPackage('transformers')
def test_vision_transformer(device):
    model = VisionTransformer(
        model_name='microsoft/swin-base-patch4-window7-224', ).to(device)
    assert model.device == device
    assert str(
        model
    ) == 'VisionTransformer(model_name=microsoft/swin-base-patch4-window7-224)'

    images = torch.randn(2, 3, 224, 224).to(device)

    out = model(images)
    assert out.device == device
    assert out.size() == (2, 49, 1024)

    out = model(images, output_device='cpu')
    assert out.is_cpu
    assert out.size() == (2, 49, 1024)
