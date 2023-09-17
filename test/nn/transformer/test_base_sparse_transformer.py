import torch

from torch_geometric.nn.transformer import SequenceModel
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.modeling_outputs import ImageClassifierOutput

def test_output():
    sequence_input = torch.randn(32, 3, 224, 224)
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
    config.cond = False
    config.order = 2 
    config.block_list = [14, 14]
    config.sparsity = 90
    config.load_from_pretrained = False 
    config.num_labels = 10
    model = SequenceModel(model="ViT", config=config, load_from_pretrained=False, self_define=False)
    output = model(sequence_input)
    assert type(output) is ImageClassifierOutput 
    logit = output['logits']
    assert logit.size() == (32, 10)