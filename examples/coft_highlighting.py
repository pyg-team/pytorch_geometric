"""
COFT Highlighting Example
=========================

This example demonstrates how to use the COFT module for
context-aware highlighting of entity surface forms within text,
leveraging graph-based recall + LLM-based scoring.

The example uses:
- A tiny knowledge graph (3 entities, 2 relations)
- A deterministic FakeLLM (no real model download required)
- Three highlight granularities (word, sentence, paragraph)

Run:
    python examples/coft_highlighting.py
"""

import torch
from torch_geometric.llm.highlighting.coft import COFT
from torch_geometric.llm.highlighting.granularity import Granularity
from torch_geometric.llm.highlighting.fake_llm import FakeLLM


# -------------------------------------------------------------------------
# Small demo KG + alias dictionary
# -------------------------------------------------------------------------
def build_demo_kg():
    triplets = [
        ("Q3", "R1", "Q1"),   # fruit ↔ apple
        ("Q1", "R1", "Q2"),   # apple ↔ banana
    ]
    alias = {
        "Q1": ["apple", "red apple"],
        "Q2": ["banana"],
        "Q3": ["fruit"],
    }
    return triplets, alias


# -------------------------------------------------------------------------
# Example execution
# -------------------------------------------------------------------------
def main():
    triplets, alias = build_demo_kg()
    coft = COFT(FakeLLM(), triplets, alias)

    query = "Tell me about fruit."
    reference = (
        "Apple is rich in fiber. "
        "Banana contains potassium. "
        "Red apple is sweet and crisp. "
        "Fruit varieties offer many benefits."
    )

    print("\n=== Original Text ===\n")
    print(reference)

    # ------------------------------------------------------------------
    # Sentence-level highlight (default)
    # ------------------------------------------------------------------
    print("\n=== Sentence-Level Highlight ===\n")
    sent_hl = coft.highlight(query, reference, granularity=Granularity.SENTENCE)
    print(sent_hl)

    # ------------------------------------------------------------------
    # Word-level highlight
    # ------------------------------------------------------------------
    print("\n=== Word-Level Highlight ===\n")
    word_hl = coft.highlight(query, reference, granularity=Granularity.WORD)
    print(word_hl)

    # ------------------------------------------------------------------
    # Paragraph-level highlight
    # ------------------------------------------------------------------
    para_text = (
        "Apple paragraph text.\n"
        "Banana paragraph text.\n"
        "Fruit paragraph text."
    )
    print("\n=== Paragraph-Level Highlight ===\n")
    para_hl = coft.highlight("fruit", para_text, granularity=Granularity.PARAGRAPH)
    print(para_hl)


if __name__ == "__main__":
    main()
