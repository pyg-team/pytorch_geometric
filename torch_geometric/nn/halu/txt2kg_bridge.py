from __future__ import annotations

from typing import Any

from .utils import triples_to_text, get_num_procs

def txt2kg_to_text(txt2kg: Any, key: Any) -> str:
    """
    Emit Evidence: text from a TXT2KG instance. If the instance exposes
    .to_text(key) we call it; otherwise we read .relevant_triples[key] and format
    to "('h','r','t')" lines (PDF Listing 7).
    """
    # Native helper available?
    to_text = getattr(txt2kg, "to_text", None)
    if callable(to_text):
        return to_text(key)
    # Fallback to formatting the triplets dict
    rel = getattr(txt2kg, "relevant_triples", None) or {}
    trips = rel.get(key, [])
    return triples_to_text(trips)


def patch_txt2kg_safeguards() -> bool:
    """
    Apply two runtime hardening patches to torch_geometric.llm.models.txt2kg:

    1) Avoid chain-of-thought forcing for "known reasoners" by clearing the
       `known_reasoners` list/tuple if present.
    2) Replace `_get_num_procs()` with a hardened version.

    Returns True if any patch was applied, False otherwise.
    """
    try:
        import torch_geometric.llm.models.txt2kg as mod
    except Exception:
        return False

    changed = False

    # (1) Neutralize CoT forcing list if present
    if hasattr(mod, "known_reasoners"):
        try:
            setattr(mod, "known_reasoners", ())
            changed = True
        except Exception:
            pass

    # (2) Patch _get_num_procs with our hardened version
    try:
        def _patched_get_num_procs():
            return get_num_procs()
        setattr(mod, "_get_num_procs", _patched_get_num_procs)
        changed = True
    except Exception:
        pass

    return changed
