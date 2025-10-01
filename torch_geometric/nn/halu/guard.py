from __future__ import annotations

import os
from typing import Optional, Any, Dict, Literal

from .detector import HallucinationDetector, build_prompt
from .utils import graph_to_text_evidence

_Mode = Literal["postgen", "pregate"]


class GuardedLLM:
    """
    Convenience wrapper for inserting the detector into a PyG GNN+LLM flow.
    Modes:
      - "pregate": assess risk first, only then run the expensive generator
      - "postgen": generate first, then assess and optionally redact/abstain

    The `generator_llm` is expected to implement one of:
      - .inference(question=[prompt], max_tokens=int) -> List[str]
      - __call__(prompt) -> str

    Optional policy knob:
      - HALU_ROH_MAX: if set (e.g., "0.25"), require roh_bound <= HALU_ROH_MAX
        in addition to the detector's decision_answer.
    """

    def __init__(
        self,
        detector: HallucinationDetector,
        generator_llm: Any,
        mode: _Mode = os.getenv("HALU_MODE", "postgen"),
    ) -> None:
        assert mode in ("postgen", "pregate"), "mode must be 'postgen' or 'pregate'"
        self.detector = detector
        self.generator = generator_llm
        self.mode: _Mode = mode  # type: ignore[assignment]
        self._roh_max = os.getenv("HALU_ROH_MAX")
        self._roh_max = float(self._roh_max) if self._roh_max not in (None, "") else None

    def __call__(
        self,
        query: str,
        data: Optional[Any] = None,
        evidence_text: Optional[str] = None,
        max_tokens: int = 512,
        return_prompt: bool = False,
    ) -> Dict[str, Any]:
        evidence = evidence_text or (graph_to_text_evidence(data) if data is not None else None)
        prompt = build_prompt(query, evidence)

        if self.mode == "pregate":
            metrics = self.detector.assess(query=query, evidence_text=evidence)
            if not self._allow(metrics):
                return {"answer": None, "abstained": True, "metrics": metrics, **({"prompt": prompt} if return_prompt else {})}
            answer = self._generate(prompt, max_tokens=max_tokens)
            return {"answer": answer, "abstained": False, "metrics": metrics, **({"prompt": prompt} if return_prompt else {})}

        # post-generation policy
        answer = self._generate(prompt, max_tokens=max_tokens)
        metrics = self.detector.assess(query=query, evidence_text=evidence)
        if not self._allow(metrics):
            return {"answer": None, "abstained": True, "metrics": metrics, "redacted": True, **({"prompt": prompt} if return_prompt else {})}
        return {"answer": answer, "abstained": False, "metrics": metrics, **({"prompt": prompt} if return_prompt else {})}

    # Internal helper
    def _generate(self, prompt: str, *, max_tokens: int) -> str:
        gen = self.generator
        if hasattr(gen, "inference"):
            out = gen.inference(question=[prompt], max_tokens=max_tokens)
            if isinstance(out, (list, tuple)) and len(out) > 0:
                return str(out[0])
            return str(out)
        if callable(gen):
            return str(gen(prompt))
        raise TypeError(
            "generator_llm must be callable or expose .inference(question=[...], max_tokens=...)"
        )

    def _allow(self, metrics: Dict[str, Any]) -> bool:
        ok = bool(metrics.get("decision_answer", False))
        if ok and self._roh_max is not None:
            ok = float(metrics.get("roh_bound", 1.0)) <= float(self._roh_max)
        return ok
