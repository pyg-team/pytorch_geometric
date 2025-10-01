from __future__ import annotations

from typing import Iterable, Tuple, Any

def graph_to_text_evidence(data: Any) -> str:
    """
    Convert a torch_geometric.data.Data (or HeteroData) into one-triple-per-line
    textual evidence like: "('h','r','t')" (see PDF Listing 9).
    Falls back gracefully if optional fields are absent.

    Expected optional attributes on `data`:
    - node_name: Tensor/List[str] mapping node indices to human-readable names
    - edge_index: (2, E) tensor-like of integer indices
    - edge_type: optional relation label per edge
    """
    name_attr = getattr(data, "node_name", None)

    def node_name(i: int) -> str:
        return str(name_attr[i]) if name_attr is not None else f"n{i}"

    lines = []
    if hasattr(data, "edge_index"):
        ei = data.edge_index
        etype = getattr(data, "edge_type", None)
        # Support torch tensors, numpy and lists
        E = int(ei.shape[1]) if hasattr(ei, "shape") else len(ei[0])
        for k in range(E):
            s = int(ei[0, k]) if hasattr(ei, "__getitem__") else int(ei[0][k])
            t = int(ei[1, k]) if hasattr(ei, "__getitem__") else int(ei[1][k])
            if etype is not None:
                try:
                    r = str(etype[k].item() if hasattr(etype[k], "item") else etype[k])
                except Exception:
                    r = str(etype[k])
            else:
                r = "linked_to"
            lines.append(f"('{node_name(s)}','{r}','{node_name(t)}')")
    return "\n".join(lines)


def triples_to_text(triples: Iterable[Tuple[str, str, str]]) -> str:
    """
    Format a list of (head, relation, tail) triples into one-line-per-triple
    representation expected by the detector:
    "('h','r','t')\n('h2','r2','t2')"
    """
    return "\n".join([f"('{h}','{r}','{t}')" for (h, r, t) in triples])


def get_num_procs() -> int:
    """
    Hardened helper (see PDF Listing 6). Uses half the available CPUs, with fallbacks.
    """
    import os

    num_proc = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_proc = len(os.sched_getaffinity(0)) // 2
        except Exception:
            num_proc = None
    if not num_proc:
        num_proc = max(1, (os.cpu_count() or 2) // 2)
    return int(num_proc)
