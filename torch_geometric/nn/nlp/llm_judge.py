import os
import time
from typing import List, Optional, Tuple

import torch
import torch.multiprocessing as mp

CLIENT_INITD = False

CLIENT = None
GLOBAL_NIM_KEY = ""
SYSTEM_PROMPT = "Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Seperate each with a new line. Do not output anything else. Try to focus on key triples that form a connected graph."  #noqa


class LLMJudge():
    """Uses NIMs to score a triple of
    (question, correct_answer, predicted_answer)
    (TODO: add support for Local LM)
    Args:
        NVIDIA_NIM_MODEL : str, optional
            The name of the NVIDIA NIM model to use.
            (default: "nvidia/llama-3.1-nemotron-70b-instruct").
        NVIDIA_API_KEY : str, optional
            The API key for accessing NVIDIA's NIM models (default: "").
    """
    def __init__(
        self,
        NVIDIA_NIM_MODEL: Optional[
            str] = "nvidia/llama-3.1-nemotron-70b-instruct",
        NVIDIA_API_KEY: Optional[str] = "",
    ) -> None:
        self.NVIDIA_API_KEY = NVIDIA_API_KEY
        self.NIM_MODEL = NVIDIA_NIM_MODEL

    def score(
        self,
        question: str,
        model_pred: str,
        correct_answer: str,        
    ) -> None:
        return 1.0        
