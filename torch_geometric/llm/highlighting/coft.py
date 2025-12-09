import re
import math
from typing import List, Dict, Set, Tuple, Iterable

import torch
from torch_geometric.llm import LLM
from .granularity import Granularity


class COFT:
    """
    Highlights contextually relevant entity surface
    forms in text, based on graph proximity + LLM scoring.
    """

    def __init__(
        self,
        llm: LLM,
        triplets: List[Tuple[str, str, str]],
        entity_alias: Dict[str, List[str]]
    ) -> None:

        self.llm = llm
        assert isinstance(self.llm, LLM), \
            "LLM instance must be of type torch_geometric.llm.LLM"

        # Store mapping qid <-> index
        nodes = sorted(entity_alias.keys())
        self.node_to_id = {qid: i for i, qid in enumerate(nodes)}
        self.id_to_node = nodes
        self.num_nodes = len(nodes)

        # Original alias
        self.qid_to_surfaces = entity_alias

        # Case-insensitive mapping surface(lowercase) -> qids
        self.surface_to_qids_lower: Dict[str, List[str]] = {}
        for qid, surfaces in entity_alias.items():
            for s in surfaces:
                s_low = s.lower()
                self.surface_to_qids_lower.setdefault(s_low, []).append(qid)

        # Construct CSR adjacency list (undirected KG)
        src_list, dst_list = [], []
        for h, _, t in triplets:
            if h in self.node_to_id and t in self.node_to_id:
                u, v = self.node_to_id[h], self.node_to_id[t]
                src_list.extend([u, v])
                dst_list.extend([v, u])

        if src_list:
            src_tensor = torch.tensor(src_list, dtype=torch.long)
            dst_tensor = torch.tensor(dst_list, dtype=torch.long)
            perm = torch.argsort(src_tensor)
            src_sorted = src_tensor[perm]
            self.dst_sorted = dst_tensor[perm]

            self.indptr = torch.zeros(self.num_nodes + 1, dtype=torch.long)
            unique_src, counts = torch.unique(src_sorted, return_counts=True)
            self.indptr[unique_src + 1] = counts
            self.indptr = torch.cumsum(self.indptr, dim=0)
        else:
            self.dst_sorted = torch.empty(0, dtype=torch.long)
            self.indptr = torch.zeros(self.num_nodes + 1, dtype=torch.long)

        self.stop_words = {
            "who", "why", "which", "what", "where", "when", "how",
            "a", "an", "the", "in", "on", "at", "for", "to", "of",
            "is", "are"
        }
        self.last_loss = 0.0

    # ---------------------------------------------------------------
    # Public highlight interface
    # ---------------------------------------------------------------
    def highlight(
        self,
        query: str,
        reference: str,
        granularity: Granularity = Granularity.SENTENCE,
    ) -> str:
        candidates = self._recall_candidates(query)
        if not candidates:
            return reference

        weights = self._score_entities(query, reference, candidates)
        return self._select_and_format(reference, weights, self.last_loss, granularity)

    # ---------------------------------------------------------------
    # Recaller (case-insensitive)
    # ---------------------------------------------------------------
    def _recall_candidates(self, text: str) -> Set[str]:
        tokens = re.findall(r"\b\w+\b", text)
        found_surface_forms: Set[str] = set()

        max_ngram = 4
        n_tokens = len(tokens)

        for n in range(1, max_ngram + 1):
            for i in range(n_tokens - n + 1):
                gram = tokens[i:i + n]
                if n == 1 and gram[0].lower() in self.stop_words:
                    continue

                surface = " ".join(gram).lower()
                if surface in self.surface_to_qids_lower:
                    found_surface_forms.add(surface)

        # Map surface -> QIDs
        initial_qids = set()
        for s in found_surface_forms:
            initial_qids.update(self.surface_to_qids_lower[s])

        # Expand neighbors (1-hop)
        init_indices = [self.node_to_id[q] for q in initial_qids]
        neighbor_indices = set()

        for idx in init_indices:
            start = int(self.indptr[idx])
            end = int(self.indptr[idx + 1])
            neighbor_indices.update(self.dst_sorted[start:end].tolist())

        all_indices = set(init_indices).union(neighbor_indices)

        # Return alias surfaces (original casing)
        candidates: Set[str] = set()
        for idx in all_indices:
            qid = self.id_to_node[idx]
            candidates.update(self.qid_to_surfaces[qid])

        return candidates

    # ---------------------------------------------------------------
    # TF-ISF (case-insensitive)
    # ---------------------------------------------------------------
    def _tf_isf(self, entity: str, sentences: List[str]) -> float:
        e_low = entity.lower()
        s_lower = [s.lower() for s in sentences]

        f_s = [s.count(e_low) for s in s_lower]
        f_eS = sum(f_s)
        N = len(s_lower)

        if f_eS == 0 or N == 0:
            return 0.0

        isf = math.log2(N / (f_eS + 1))
        scores = []

        for sent_low, cnt in zip(s_lower, f_s):
            if cnt == 0:
                continue
            tf = cnt / max(1, len(sent_low.split()))
            scores.append(tf * isf)

        return sum(scores) / len(scores) if scores else 0.0

    # ---------------------------------------------------------------
    # Relevance scoring (unchanged)
    # ---------------------------------------------------------------
    def _get_entity_relevance(self, query: str, entity: str) -> float:
        loss = self.llm(question=[query], answer=[entity])
        val = float(loss.item()) if hasattr(loss, "item") else float(loss)
        return max(val, 0.0) / math.log(2)

    def _update_global_loss(self, query: str, reference: str):
        loss = self.llm(question=[query], answer=[reference])
        self.last_loss = float(loss.item()) if hasattr(loss, "item") else float(loss)

    # ---------------------------------------------------------------
    # Entity scoring (case-insensitive existence check)
    # ---------------------------------------------------------------
    def _score_entities(self, query: str, reference: str, entities: Iterable[str]) -> Dict[str, float]:

        self._update_global_loss(query, reference)
        reference_lower = reference.lower()

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", reference)
            if s.strip()
        ]

        weights = {}

        for e in entities:
            if e.lower() not in reference_lower:
                continue

            score_tf = self._tf_isf(e, sentences)
            if score_tf == 0:
                continue

            rel = self._get_entity_relevance(query, e)
            weights[e] = score_tf * rel

        return weights

    # ---------------------------------------------------------------
    # Selector & Formatter (case-insensitive)
    # ---------------------------------------------------------------
    def _select_and_format(
        self,
        reference: str,
        weights: Dict[str, float],
        scorer_loss: float,
        granularity: Granularity,
    ) -> str:

        if not weights:
            return reference

        # Dynamic threshold tau
        min_len, max_len = 200, 3000
        min_info, max_info = 1.0, 6.0
        length = len(reference.split())

        def clip(v, lo, hi):
            return max(lo, min(v, hi))

        tau_len = clip((length - min_len) / (max_len - min_len), 0, 1)
        tau_info = clip((scorer_loss - min_info) / (max_info - min_info), 0, 1)
        tau = 0.5 * (tau_len + tau_info)

        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        k = max(1, int(len(sorted_items) * tau))
        selected = {ent for ent, _ in sorted_items[:k]}

        # WORD LEVEL (regex, keep original casing)
        if granularity == Granularity.WORD:
            for e in sorted(selected, key=len, reverse=True):
                pattern = r"\b" + re.escape(e) + r"\b"
                reference = re.sub(
                    pattern,
                    lambda m: f"**{m.group(0)}**",
                    reference,
                    flags=re.IGNORECASE,
                )
            return reference

        # SENTENCE LEVEL
        elif granularity == Granularity.SENTENCE:
            parts = re.split(r"(?<=[.!?])\s+", reference)
            return " ".join(
                f"**{s}**"
                if any(e.lower() in s.lower() for e in selected)
                else s
                for s in parts
            )

        # PARAGRAPH LEVEL
        elif granularity == Granularity.PARAGRAPH:
            parts = reference.split("\n")
            return "\n".join(
                f"**{p}**"
                if any(e.lower() in p.lower() for e in selected)
                else p
                for p in parts
            )

        return reference
