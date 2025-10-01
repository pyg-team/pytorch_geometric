from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# We import from the attached hallucination toolkit (hallbayes).
# If it is not available at runtime, we raise a helpful error message when assess() is called.
_TOOLKIT_IMPORT_ERROR: Optional[Exception] = None
try:  # defer hard failure to first usage so that type checkers and docs build fine
    from hallucination_toolkit import OpenAIBackend, OpenAIPlanner, OpenAIItem
except Exception as e:  # pragma: no cover - exercised in environments w/o toolkit
    _TOOLKIT_IMPORT_ERROR = e  # keep for later; we don't fail import of this module


def _require_toolkit():  # pragma: no cover - exercised only when missing
    if _TOOLKIT_IMPORT_ERROR is not None:
        raise ImportError(
            "hallucination_toolkit not found. Please vendor or install "
            "https://github.com/leochlon/hallbayes/blob/main/scripts/hallucination_toolkit.py "
            "on your PYTHONPATH. Original import error: %r" % (_TOOLKIT_IMPORT_ERROR,)
        )


# Environment-backed defaults
_DEFAULT_MODEL = os.getenv("HALU_MODEL", "gpt-4o-mini")
_DEFAULT_ISR_THRESHOLD = float(os.getenv("ISR_THRESHOLD", "1.0"))
_DEFAULT_MARGIN_BITS = float(os.getenv("MARGIN_BITS", "0.0"))
_DEFAULT_H_STAR = float(os.getenv("H_STAR", "0.05"))
_DEFAULT_SKELETON_POLICY = os.getenv("SKELETON_POLICY", "auto")

@dataclass
class RiskMetrics:
    decision_answer: bool
    delta_bar: float
    b2t: float
    isr: float
    roh_bound: float
    rationale: str
    closed_book: bool
    q_avg: float
    q_conservative: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HallucinationDetector:
    """
    Hallucination risk detector that wraps the OpenAI EDFL/Δ/B2T/ISR/RoH evaluator
    from the hallbayes `hallucination_toolkit.py` script.

    Parameters follow the "Minimal Drop-in Code" in the PDF.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        isr_threshold: float = _DEFAULT_ISR_THRESHOLD,
        margin_bits: float = _DEFAULT_MARGIN_BITS,
        h_star: float = _DEFAULT_H_STAR,
        skeleton_policy: str = _DEFAULT_SKELETON_POLICY,
    ) -> None:
        self.model = str(model)
        self.isr_threshold = float(isr_threshold)
        self.margin_bits = float(margin_bits)
        self.h_star = float(h_star)
        self.skeleton_policy = str(skeleton_policy)

    def assess(self, *, query: str, evidence_text: Optional[str]) -> Dict[str, Any]:
        """
        Score a (query, optional evidence) pair and return the risk metrics as a dict.
        Keys: decision_answer, delta_bar, b2t, isr, roh_bound, rationale, closed_book,
        q_avg, q_conservative.

        This function uses skeleton_policy="auto" to automatically select evidence-erase
        or closed-book skeletons based on whether an 'Evidence:' field is present.
        """
        _require_toolkit()

        prompt = build_prompt(query, evidence_text)
        item = OpenAIItem(
            prompt=prompt,
            skeleton_policy=self.skeleton_policy,  # "auto" by default
            fields_to_erase=["Evidence"],
            n_samples=3,
            m=6,
        )
        backend = OpenAIBackend(model=self.model)
        planner = OpenAIPlanner(
            backend, temperature=0.5, max_tokens_decision=8, q_floor=None
        )

        m = planner.evaluate_item(
            idx=0,
            item=item,
            h_star=self.h_star,
            isr_threshold=self.isr_threshold,
            margin_extra_bits=self.margin_bits,
            B_clip=12.0,
            clip_mode="one-sided",
        )

        out = RiskMetrics(
            decision_answer=bool(m.decision_answer),
            delta_bar=float(m.delta_bar),
            b2t=float(m.b2t),
            isr=float(m.isr),
            roh_bound=float(m.roh_bound),
            rationale=str(m.rationale),
            closed_book=bool(getattr(m.meta, "get", lambda *_: False)("closed_book", False) if hasattr(m, "meta") else getattr(m, "closed_book", False)),
            q_avg=float(getattr(m, "q_avg", 0.0)),
            q_conservative=float(getattr(m, "q_conservative", 0.0)),
        )
        return out.to_dict()


def build_prompt(query: str, evidence: Optional[str]) -> str:
    """
    Unified prompt builder used for both risk scoring and answer generation. If `evidence`
    is provided, an 'Evidence:' section is embedded so the toolkit's skeletonizers can
    perform evidence-erase; otherwise closed-book is used automatically.
    """
    if evidence:
        return (
            "Task: Answer the user's question using the EVIDENCE below.\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{evidence}\n"
        )
    return f"Task: Answer the user's question.\nQuestion: {query}\n"
