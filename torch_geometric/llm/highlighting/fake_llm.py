import torch
import torch.nn as nn
from torch_geometric.llm import LLM

class FakeLLM(LLM):
    """
    A minimal fake LLM that returns deterministic numeric scores.
    Loss = 0.1 * len(answer[0])
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = "fake-llm"

    def forward(self, question, answer):
        # deterministic fake score
        return torch.tensor(0.1 * len(answer[0]), dtype=torch.float32)
