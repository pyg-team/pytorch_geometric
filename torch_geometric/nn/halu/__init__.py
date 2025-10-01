"""
torch_geometric.nn.halu
-----------------------
Integration layer that wires the *hallucination_toolkit* (from
https://github.com/leochlon/hallbayes/blob/main/scripts/hallucination_toolkit.py)
into a PyTorch Geometric (PyG) GNN+LLM pipeline.

Exports:
- HallucinationDetector: risk-scoring engine (OpenAI backend)
- build_prompt: unified prompt builder for risk + generation
- graph_to_text_evidence: textify a Data/HeteroData graph
- GuardedLLM: convenience wrapper for pre-gate/post-gen flows
- triples_to_text: turn list of (h,r,t) into Evidence: text
- txt2kg_to_text: emit Evidence: text from an instance of TXT2KG
- patch_txt2kg_safeguards: apply hardening patches to TXT2KG at runtime
"""
from .detector import HallucinationDetector, build_prompt
from .utils import graph_to_text_evidence, triples_to_text, get_num_procs
from .guard import GuardedLLM
from .txt2kg_bridge import txt2kg_to_text, patch_txt2kg_safeguards

__all__ = [
    "HallucinationDetector",
    "build_prompt",
    "graph_to_text_evidence",
    "GuardedLLM",
    "triples_to_text",
    "get_num_procs",
    "txt2kg_to_text",
    "patch_txt2kg_safeguards",
]
