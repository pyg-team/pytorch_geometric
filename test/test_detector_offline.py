import sys
from types import SimpleNamespace
import importlib
from pathlib import Path

# Inject a fake 'hallucination_toolkit' so we can test offline without OpenAI.
class _FakeItem:
    def __init__(self, prompt, skeleton_policy="auto", fields_to_erase=None, n_samples=1, m=1):
        self.prompt = prompt
        self.skeleton_policy = skeleton_policy
        self.fields_to_erase = fields_to_erase or []
        self.n_samples = n_samples
        self.m = m

class _FakeBackend:
    def __init__(self, model="gpt-4o-mini"): self.model = model

class _FakeEval:
    def __init__(self, decision_answer, delta_bar, b2t, isr, roh_bound, rationale, closed_book, q_avg, q_conservative):
        self.decision_answer = decision_answer
        self.delta_bar = delta_bar
        self.b2t = b2t
        self.isr = isr
        self.roh_bound = roh_bound
        self.rationale = rationale
        self.meta = {"closed_book": closed_book}
        self.q_avg = q_avg
        self.q_conservative = q_conservative

class _FakePlanner:
    def __init__(self, backend, temperature=0.5, max_tokens_decision=8, q_floor=None): pass
    def evaluate_item(self, idx, item, h_star, isr_threshold, margin_extra_bits, B_clip=12.0, clip_mode="one-sided"):
        has_evidence = "Evidence:" in item.prompt
        # Toy monotonic behavior: evidence increases delta and ISR, flips decision
        b2t = 1.0
        delta = 1.2 if has_evidence else 0.4
        isr = delta / b2t
        decision = isr >= isr_threshold
        return _FakeEval(decision, delta, b2t, isr, roh_bound=max(0.0, 1.0-isr), rationale="fake", closed_book=not has_evidence, q_avg=0.5, q_conservative=0.3)

# Prepare the fake module
sys.modules["hallucination_toolkit"] = SimpleNamespace(
    OpenAIBackend=_FakeBackend, OpenAIPlanner=_FakePlanner, OpenAIItem=_FakeItem
)

# Import our package
spec = importlib.util.spec_from_file_location(
    "halu_detector",
    str(Path(__file__).resolve().parents[1] / "torch_geometric" / "nn" / "halu" / "detector.py")
)
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)

def test_build_prompt_closed_book():
    s = detector.build_prompt("Q?", None)
    assert "Evidence:" not in s

def test_build_prompt_with_evidence():
    s = detector.build_prompt("Q?", "('a','r','b')")
    assert "Evidence:" in s

def test_assess_increases_isr_with_evidence(tmp_path):
    det = detector.HallucinationDetector(isr_threshold=1.0)
    m_no = det.assess(query="Q?", evidence_text=None)
    m_yes = det.assess(query="Q?", evidence_text="('a','r','b')")
    assert m_yes["isr"] > m_no["isr"]
    # Decision True only when evidence is present in our fake
    assert m_no["decision_answer"] is False
    assert m_yes["decision_answer"] is True
