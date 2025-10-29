import os

import pytest
import torch

from torch_geometric.llm.models.txt2kg import (
    _chunk_text,
    _llm_then_python_parse,
    _multiproc_helper,
    _parse_n_check_triples,
)


@pytest.mark.parametrize("text, chunk_size, expected", [
    ("short text", 100, ["short text"]),
    ("", 10, []),
])
def test_chunk_text_short(text, chunk_size, expected):
    assert _chunk_text(text, chunk_size=chunk_size) == expected


@pytest.mark.parametrize("text", [
    "Hello world. This is a test! And another sentence?",
    "Sentence one. Sentence two. Sentence three.",
])
def test_chunk_text_general(text):
    chunks = _chunk_text(text, chunk_size=20)
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)
    assert len(" ".join(chunks)) >= len(text) * 0.8


@pytest.mark.parametrize("triples_str", [
    "('A','rel','B')\n('C','rel','D')",
    "(A,rel,B) (C,rel,D)",
])
def test_parse_n_check_triples_strict(triples_str):
    triples = _parse_n_check_triples(triples_str)
    assert ("A", "rel", "B") in triples
    assert ("C", "rel", "D") in triples


def test_llm_then_python_parse():
    chunks = ["chunk1", "chunk2"]

    def fake_llm_fn(chunk, **kwargs):
        return f"('{chunk}','rel','X')"

    def fake_py_fn(txt):
        # 模拟解析函数
        return [("A", "rel", "B")]

    res = _llm_then_python_parse(chunks, fake_py_fn, fake_llm_fn)
    assert isinstance(res, list)
    assert all(len(t) == 3 for t in res)


def test_multiproc_helper_creates_output_file():
    rank = 0
    out_path = f"/tmp/outs_for_proc_{rank}"
    if os.path.exists(out_path):
        os.remove(out_path)

    # 纯函数依赖，不用mock
    def fake_py_fn(x):
        return [("A", "rel", "B")]

    def fake_llm_fn(chunk, **kwargs):
        return "('x','r','y')"

    _multiproc_helper(
        rank=rank,
        in_chunks_per_proc={0: ["abc"]},
        py_fn=fake_py_fn,
        llm_fn=fake_llm_fn,
        NIM_KEY="key",
        NIM_MODEL="model",
        ENDPOINT_URL="url",
    )

    assert os.path.exists(out_path)
    out = torch.load(out_path)
    assert isinstance(out, list)
    assert all(len(t) == 3 for t in out)
    os.remove(out_path)


# @onlyOnline
# @withPackage("transformers")
# def test_chunk_to_triples_str_local_no_external_import():
#     class FakeLLM:
#         def __init__(self, name):
#             self.name = name

#         def eval(self):
#             return self

#         def inference(self, question, max_tokens):
#             return ["('A','B','C')"]

#     fake_module = types.SimpleNamespace(LLM=FakeLLM)
#     sys.modules["torch_geometric.nn.nlp"] = fake_module

#     model = TXT2KG(local_LM=True)
#     out = model._chunk_to_triples_str_local("This is a test text.")
#     assert isinstance(out, str)
#     assert "A" in out
#     assert model.total_chars_parsed > 0
#     assert model.time_to_parse > 0

# @onlyOnline
# @withPackage("transformers")
# def test_add_doc_and_save():
#     model = TXT2KG(local_LM=True)

#     model.add_doc_2_KG("graph knowledge text", QA_pair=("Q1", "A1"))
#     assert ("Q1", "A1") in model.relevant_triples
#     assert model.relevant_triples[("Q1", "A1")] == [("a", "b", "c")]

#     # 保存 + 加载
#     save_path = "/tmp/kg.pt"
#     model.save_kg(save_path)
#     loaded = torch.load(save_path)
#     assert loaded["Q1", "A1"] == [("a", "b", "c")]
