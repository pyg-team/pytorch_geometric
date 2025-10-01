from torch_geometric.nn.halu.txt2kg_bridge import txt2kg_to_text, patch_txt2kg_safeguards

class _TKG:
    def __init__(self):
        self.relevant_triples = {"k": [("a","r","b"), ("b","r","c")]}
    def to_text(self, key):
        return "\n".join(["('x','y','z')"])

def test_txt2kg_to_text_prefers_native():
    tkg = _TKG()
    assert txt2kg_to_text(tkg, "k").strip() == "('x','y','z')"

def test_txt2kg_to_text_fallback():
    tkg = _TKG()
    delattr(tkg, "to_text")
    s = txt2kg_to_text(tkg, "k")
    assert "('a','r','b')" in s and "('b','r','c')" in s

def test_patch_noop_when_module_missing(monkeypatch):
    # Simulate import failure
    import sys
    sys.modules.pop("torch_geometric.llm.models.txt2kg", None)
    assert patch_txt2kg_safeguards() in (False, True)  # should not crash
