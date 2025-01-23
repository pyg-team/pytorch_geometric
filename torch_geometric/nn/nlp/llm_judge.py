import os
import time
from typing import List, Optional, Tuple

import torch
import torch.multiprocessing as mp


# Credit for system prompts goes to Gilberto Titericz (NVIDIA)
SYSTEM_PROMPT_1 = (
    "Instruction: You are a world class state of the art "
    + "assistant for rating "
    + "a User Answer given a Question. The Question is completely"
    + " answered by the Reference Answer.\n"
    + "Say 4, if User Answer is full contained and equivalent to"
    + " Reference Answer"
    + "in all terms, topics, numbers, metrics, dates and units.\n"
    + "Say 2, if User Answer is partially contained and almost "
    + "equivalent to Reference Answer"
    + "in all terms, topics, numbers, metrics, dates and units.\n"
    + "Say 0, if User Answer is not contained in Reference Answer"
    + " or not accurate in all terms, topics,"
    + "numbers, metrics, dates and units or the User Answer do not"
    + " answer the question.\n"
    + "Do not explain or justify your rating. Your rating must be "
    + "only 4, 2 or 0 according to the instructions above.\n"
    + "### Question: {question}\n"
    + "### {answer0}: {model_pred}\n"
    + "### {answer1}: {correct_answer}\n"
    + "The rating is:\n"
)

SYSTEM_PROMPT_2 = (
    + "I will rate the User Answer in comparison to the Reference "
    + "Answer for a given Question.\n"
    + "A rating of 4 indicates that the User Answer is entirely "
    + "consistent with the Reference Answer, covering all aspects,"
    + " topics, numbers, metrics, dates, and units.\n"
    + "A rating of 2 signifies that the User Answer is mostly "
    + "aligned with the Reference Answer, with minor discrepancies"
    + " in some areas.\n"
    + "A rating of 0 means that the User Answer is either "
    + "inaccurate, incomplete, or unrelated to the Reference "
    + "Answer, or it fails to address the Question.\n"
    + "I will provide the rating without any explanation or "
    + "justification, adhering to the following scale: "
    + "0 (no match), 2 (partial match), 4 (exact match).\n"
    + "Do not explain or justify my rating. My rating must"
    + " be only 4, 2 or 0 only.\n\n"
    + "Question: {query}\n\n"
    + "{answer0}: {sentence_inference}\n\n"
    + "{answer1}: {sentence_true}\n\n"
    + "Rating: "
)


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
    ):
        """
        Args:
            question (str): The original question asked to the model.
            model_pred (str): The prediction made by the model.
            correct_answer (str): The actual correct answer to the question.

        Returns:
            None
        """
        #prompt1 = SYSTEM_PROMPT_1.format(query=question, sentence_inference=model_pred, )


        return 1.0        
