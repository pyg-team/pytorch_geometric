from typing import Mapping

import torch
from torch.nn import Parameter

from torch_geometric.nn.parameter_dict import ParameterDict


def test_internal_external_key_conversion():
    assert ParameterDict.to_internal_key("a.b") == "a#b"
    assert ParameterDict.to_internal_key("ab") == "ab"
    assert ParameterDict.to_internal_key("a.b.c") == "a#b#c"

    assert ParameterDict.to_external_key("a#b") == "a.b"
    assert ParameterDict.to_external_key("a#b#c") == "a.b.c"


def test_dot_syntax_keys():
    parameters: Mapping[str, Parameter] = {
        "param1": Parameter(torch.Tensor(16, 16)),
        "model.param2": Parameter(torch.Tensor(8, 8)),
        "model.sub_model.param3": Parameter(torch.Tensor(4, 4)),
    }
    parameter_dict = ParameterDict(parameters)

    expected_keys = {"param1", "model.param2", "model.sub_model.param3"}
    assert set(parameter_dict.keys()) == expected_keys

    for key in expected_keys:
        assert key in parameter_dict

    del parameter_dict["model.param2"]
    assert "model.param2" not in parameter_dict
