"""
TXT2KG → Guarded RAG example
----------------------------
Build triples with TXT2KG, convert them to Evidence: text, then run the risk
detector and (conditionally) generate an answer.

This script demonstrates:
  1) Optional TXT2KG hardening (no chain-of-thought; robust _get_num_procs)
  2) Converting TXT2KG triples to Evidence: text
  3) Using the GuardedLLM wrapper in "pregate" or "postgen" mode

It assumes `hallucination_toolkit.py` is available and that your OpenAI credentials
are exported as required by the toolkit.
"""
from typing import List, Tuple

try:
    # TXT2KG is optional; we fall back to a tiny stub so the file remains runnable.
    from torch_geometric.llm.models.txt2kg import TXT2KG  # type: ignore
    HAS_TXT2KG = True
except Exception:  # pragma: no cover
    HAS_TXT2KG = False
    class TXT2KG:  # minimal stub for offline demo
        def __init__(self, *_, **__):
            self.relevant_triples = {}
        def add_doc_2_KG(self, text: str, QA_pair=None):
            k = QA_pair or "default"
            self.relevant_triples[k] = [("alpha","linked_to","beta"), ("beta","linked_to","gamma")]
        def to_text(self, key) -> str:
            trips = self.relevant_triples.get(key, [])
            return "\n".join([f"('{h}','{r}','{t}')" for (h,r,t) in trips])

try:
    import torch
except Exception:  # pragma: no cover
    class torch:
        @staticmethod
        def manual_seed(_): pass

from torch_geometric.nn.halu import (
    HallucinationDetector, GuardedLLM, build_prompt,
    txt2kg_to_text, patch_txt2kg_safeguards
)

class DummyGen:
    def inference(self, question: List[str], max_tokens: int = 128):
        # Echo the prompt (replace with your real generator).
        return [f"[dummy-answer] {question[0][:80]} ..."]

def main():
    # (1) Hardening patches for TXT2KG (no CoT forcing; robust _get_num_procs)
    if HAS_TXT2KG:
        patched = patch_txt2kg_safeguards()
        print(f"TXT2KG patched: {patched}")

    # (2) Build triples with TXT2KG from a snippet of text
    txt = "Alpha is directly related to Beta. Beta is related to Gamma."
    q = "Which node is connected to 'alpha'?"
    key = (q, None)  # follow TXT2KG style key; any hashable key is fine

    tkg = TXT2KG(local_LM=True)  # or configure for your cloud path
    tkg.add_doc_2_KG(txt, QA_pair=key)

    evidence = txt2kg_to_text(tkg, key)
    print("Evidence from TXT2KG:\n", evidence)

    # (3) Run detector + guarded generation
    det = HallucinationDetector()                # thresholds via env or args
    gen = DummyGen()                             # replace with your generator
    guard = GuardedLLM(detector=det, generator_llm=gen, mode="pregate")
    out = guard(q, evidence_text=evidence)

    print("Risk metrics:", out["metrics"])
    print("Guarded answer:", out["answer"])

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
