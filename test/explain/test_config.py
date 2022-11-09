import pytest

from torch_geometric.explain.config import ExplainerConfig, ThresholdConfig


@pytest.mark.parametrize('threshold_pairs', [
    ('hard', 0.5, True),
    ('hard', 1.1, False),
    ('hard', -1, False),
    ('topk', 1, True),
    ('topk', 0, False),
    ('topk', -1, False),
    ('topk', 0.5, False),
    ('invalid', None, False),
    ('hard', None, False),
])
def test_threshold_config(threshold_pairs):
    threshold_type, threshold_value, valid = threshold_pairs
    if valid:
        threshold = ThresholdConfig(threshold_type, threshold_value)
        assert threshold.type.value == threshold_type
        assert threshold.value == threshold_value
    else:
        with pytest.raises(ValueError):
            ThresholdConfig(threshold_type, threshold_value)


@pytest.mark.parametrize('explanation_type', [
    'model',
    'phenomenon',
    'invalid',
])
@pytest.mark.parametrize('mask_type', [
    None,
    'object',
    'attributes',
    'invalid',
])
def test_configuration_config(explanation_type, mask_type):
    if (explanation_type != 'invalid' and mask_type is not None
            and mask_type != 'invalid'):
        config = ExplainerConfig(explanation_type, mask_type, mask_type)
        assert config.explanation_type.value == explanation_type
        assert config.node_mask_type.value == mask_type
        assert config.edge_mask_type.value == mask_type
    else:
        with pytest.raises(ValueError):
            ExplainerConfig(explanation_type, mask_type, mask_type)
