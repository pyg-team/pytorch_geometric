import re
import math
from typing import List, Dict, Set, Tuple, Iterable

import torch
from torch_geometric.llm import LLM
from enum import Enum


class Granularity(Enum):
    WORD = "word"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


class COFT:
    r"""The COFT model from the `"Coarse-to-Fine Highlighting:
    Reducing Knowledge Hallucination in Large Language Models"
    <https://arxiv.org/pdf/2410.15116>`_ paper.

    Args:
        llm (LLM): The LLM instance to use.
        triplets (Iterable[Tuple[str, str, str]]): The knowledge graph triplets
            in the format (head, relation, tail).
        entity_alias (Dict[str, Iterable[str]]): A dictionary mapping entity IDs
            to their surface form aliases.

    .. note::
        For an example of using :class:`COFT`, see
        `examples/llm/coft.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/llm/coft.py>`_.
    """

    def __init__(self, llm: LLM,
                 triplets: List[Tuple[str, str, str]],
                 entity_alias: Dict[str, List[str]]):

        self.llm = llm
        assert isinstance(self.llm, LLM)

        # Mapping qid <-> id
        nodes = sorted(entity_alias.keys())
        self.node_to_id = {qid: i for i, qid in enumerate(nodes)}
        self.id_to_node = nodes
        self.num_nodes = len(nodes)

        self.qid_to_surfaces = entity_alias

        # surface → qids (lowercased)
        self.surface_to_qids_lower = {}
        for qid, surfaces in entity_alias.items():
            for s in surfaces:
                self.surface_to_qids_lower.setdefault(s.lower(), []).append(qid)

        # Build adjacency CSR
        src = []
        dst = []
        for h, _, t in triplets:
            if h in self.node_to_id and t in self.node_to_id:
                u = self.node_to_id[h]
                v = self.node_to_id[t]
                src.extend([u, v])
                dst.extend([v, u])

        if src:
            src = torch.tensor(src, dtype=torch.long)
            dst = torch.tensor(dst, dtype=torch.long)
            perm = torch.argsort(src)
            src_sorted = src[perm]
            self.dst_sorted = dst[perm]

            self.indptr = torch.zeros(self.num_nodes + 1, dtype=torch.long)
            uniq, counts = torch.unique(src_sorted, return_counts=True)
            self.indptr[uniq + 1] = counts
            self.indptr = torch.cumsum(self.indptr, dim=0)
        else:
            self.dst_sorted = torch.empty(0)
            self.indptr = torch.zeros(self.num_nodes + 1)

        self.stop_words = {
            "the", "is", "are", "a", "an",
            "what", "when", "where", "who", "how",
            "of", "in", "on", "at", "for", "to"
        }

        self.last_loss = 0.0

    # ================================================================
    # Public API
    # ================================================================
    def highlight(self, query: str, reference: str,
                  granularity: Granularity = Granularity.SENTENCE,
                  selector_cfg=None
    ) -> str:
        r"""The highlight pass to select important text segments.

        Args:
            query (str): The input question or prompt.
            reference (str): The text to highlight.
            granularity (Granularity, optional): The level of highlighting
                granularity (WORD, SENTENCE, or PARAGRAPH).
                (default: :obj:`Granularity.SENTENCE`)
            min_len (int, optional): The minimum text length threshold for
                scaling the selection ratio. (default: :obj:`200`)
            max_len (int, optional): The maximum text length threshold for
                scaling the selection ratio. (default: :obj:`3000`)
            min_info (float, optional): The minimum information (loss)
                threshold. (default: :obj:`1.0`)
            max_info (float, optional): The maximum information (loss)
                threshold. (default: :obj:`6.0`)
        """

        default_cfg = {
            "min_len": 0,
            "max_len": 3000,
            "min_info": 0.0,
            "max_info": 6.0,
        }
        if selector_cfg:
            default_cfg.update(selector_cfg)
        self.selector_cfg = default_cfg

        candidates = self._recall_candidates(query)
        if not candidates:
            return reference

        weights = self._score_entities(query, reference, candidates)
        return self._select_and_format(reference, weights,
                                       self.last_loss, granularity)

    # ================================================================
    # Phase 1: Recaller
    # ================================================================
    def _recall_candidates(self, text: str) -> Set[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        found = set()

        # token n-grams up to 4
        for n in range(1, 5):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i:i+n])
                if n == 1 and gram in self.stop_words:
                    continue
                if gram in self.surface_to_qids_lower:
                    found.add(gram)

        init_qids = set()
        for s in found:
            init_qids.update(self.surface_to_qids_lower[s])

        init_idx = [self.node_to_id[q] for q in init_qids]
        neigh_idx = set()

        for idx in init_idx:
            start = int(self.indptr[idx])
            end = int(self.indptr[idx+1])
            neigh_idx.update(self.dst_sorted[start:end].tolist())

        all_idx = set(init_idx).union(neigh_idx)

        result = set()
        for idx in all_idx:
            qid = self.id_to_node[idx]
            result.update(self.qid_to_surfaces[qid])
        return result

    # ================================================================
    # Phase 2: TF-ISF
    # ================================================================
    def _count_phrase(self, tokens, phrase_tokens):
        L = len(tokens)
        P = len(phrase_tokens)
        return sum(
            1 for i in range(L - P + 1)
            if tokens[i:i+P] == phrase_tokens
        )

    def _tf_isf(self, entity: str, sentences: List[str]) -> float:
        e_tokens = entity.lower().split()
        N = len(sentences)

        f_eS = 0
        tf_vals = []

        for s in sentences:
            tokens = s.lower().split()
            cnt = self._count_phrase(tokens, e_tokens)
            if cnt > 0:
                f_eS += 1
                tf_vals.append(cnt / len(tokens))

        if f_eS == 0:
            return 0.0

        isf = math.log2(N / (f_eS + 1))
        return (sum(tf_vals) / len(tf_vals)) * isf

    # ================================================================
    # Phase 3: LLM relevance
    # ================================================================
    def _get_entity_relevance(self, query: str, entity: str) -> float:
        loss = self.llm(question=[query], answer=[entity])
        val = float(loss.item())
        return math.exp(-val)  # normalized relevance

    def _update_global_loss(self, query: str, reference: str):
        loss = self.llm(question=[query], answer=[reference])
        self.last_loss = float(loss.item())

    # ================================================================
    # Phase 4: Score entities
    # ================================================================
    def _token_exists(self, ref_tokens, e_tokens):
        L = len(ref_tokens)
        P = len(e_tokens)
        return any(
            ref_tokens[i:i+P] == e_tokens
            for i in range(L - P + 1)
        )

    def _score_entities(self, query: str, reference: str, entities):

        self._update_global_loss(query, reference)

        sentences = [
            s.strip()
            for s in re.split(r"[.!?]\s*", reference)
            if s.strip()
        ]

        ref_tokens = reference.lower().split()
        weights = {}

        for e in entities:
            e_tokens = e.lower().split()

            if not self._token_exists(ref_tokens, e_tokens):
                continue

            score_tf = self._tf_isf(e, sentences)
            if score_tf == 0:
                continue

            rel = self._get_entity_relevance(query, e)
            weights[e] = score_tf * rel

        return weights

    # ================================================================
    # Highlight helpers
    # ================================================================
    def _sentence_contains(self, sentence: str, entity: str):
        s_tokens = sentence.lower().replace("*", "").split()
        e_tokens = entity.lower().split()
        return self._token_exists(s_tokens, e_tokens)

    # === non-destructive word-level highlight ===
    def _highlight_word_level(self, reference, selected):
        matches = []

        for e in selected:
            pattern = re.compile(re.escape(e), flags=re.IGNORECASE)
            for m in pattern.finditer(reference):
                matches.append((m.start(), m.end(), m.group(0)))

        matches.sort(key=lambda x: x[0])

        out = []
        last = 0
        for start, end, txt in matches:
            out.append(reference[last:start])
            out.append(f"**{txt}**")
            last = end
        out.append(reference[last:])

        return "".join(out)

    # ================================================================
    # Phase 5: Selector + formatting
    # ================================================================
    def _select_and_format(self, reference, weights, scorer_loss, granularity):

        if not weights:
            return reference

        cfg = self.selector_cfg
        L = len(reference.split())

        def clip(v, lo, hi):
            return max(lo, min(v, hi))

        tau_len = clip((L - cfg["min_len"]) / (cfg["max_len"] - cfg["min_len"]), 0, 1)
        tau_info = clip((scorer_loss - cfg["min_info"]) / (cfg["max_info"] - cfg["min_info"]), 0, 1)
        tau = 0.5 * (tau_len + tau_info)

        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        k = max(2, int(len(sorted_items) * tau))
        selected = {ent for ent, _ in sorted_items[:k]}

        # === Word level (fixed) ===
        if granularity == Granularity.WORD:
            return self._highlight_word_level(reference, selected)

        # === Sentence level ===
        if granularity == Granularity.SENTENCE:
            parts = re.split(r"(?<=[.!?])\s+", reference)
            return " ".join(
                f"**{s}**"
                if any(self._sentence_contains(s, e) for e in selected)
                else s
                for s in parts
            )

        # === Paragraph level ===
        if granularity == Granularity.PARAGRAPH:
            parts = reference.split("\n")
            return "\n".join(
                f"**{p}**"
                if any(self._sentence_contains(p, e) for e in selected)
                else p
                for p in parts
            )

        return reference
