from math import isnan
from typing import Optional

from torch_geometric.llm.models.txt2kg import \
    _chunk_to_triples_str_cloud as call_NIM

# Credit for original "Marlin Accuracy" system goes to:
# Gilberto Titericz (NVIDIA)
# This work is an adaptation of his for PyG
SYSTEM_PROMPT_1 = (
    "Instruction: You are a world class state of the art " +
    "assistant for rating " +
    "a User Answer given a Question. The Question is completely" +
    " answered by the Reference Answer.\n" +
    "Say 4, if User Answer is full contained and equivalent to" +
    " Reference Answer" +
    "in all terms, topics, numbers, metrics, dates and units.\n" +
    "Say 2, if User Answer is partially contained and almost " +
    "equivalent to Reference Answer" +
    "in all terms, topics, numbers, metrics, dates and units.\n" +
    "Say 0, if User Answer is not contained in Reference Answer" +
    " or not accurate in all terms, topics," +
    "numbers, metrics, dates and units or the User Answer do not" +
    " answer the question.\n" +
    "Do not explain or justify your rating. Your rating must be " +
    "only 4, 2 or 0 according to the instructions above.\n" +
    "### Question: \"{question}\"\n" + "### User Answer: \"{model_pred}\"\n" +
    "### Reference Answer: \"{correct_answer}\"\n" + "The rating is:\n")

SYSTEM_PROMPT_2 = (
    "I will rate the User Answer in comparison to the Reference " +
    "Answer for a given Question.\n" +
    "A rating of 4 indicates that the User Answer is entirely " +
    "consistent with the Reference Answer, covering all aspects," +
    " topics, numbers, metrics, dates, and units.\n" +
    "A rating of 2 signifies that the User Answer is mostly " +
    "aligned with the Reference Answer, with minor discrepancies" +
    " in some areas.\n" +
    "A rating of 0 means that the User Answer is either " +
    "inaccurate, incomplete, or unrelated to the Reference " +
    "Answer, or it fails to address the Question.\n" +
    "I will provide the rating without any explanation or " +
    "justification, adhering to the following scale: " +
    "0 (no match), 2 (partial match), 4 (exact match).\n" +
    "Do not explain or justify my rating. My rating must" +
    " be only 4, 2 or 0 only.\n\n" + "Question: \"{question}\"\n\n" +
    "Reference Answer: \"{model_pred}\"\n\n" +
    "User Answer: \"{correct_answer}\"\n\n" + "Rating: ")


# TODO: add support for Local LM
# TODO: add multiproc support like txt2kg
class LLMJudge():
    """Uses NIMs to score a triple of (question, model_pred, correct_answer)
    This whole class is an adaptation of Gilberto's work for PyG.

    Args:
        NVIDIA_NIM_MODEL : (str, optional)
            The name of the NVIDIA NIM model to use.
            (default: "nvidia/llama-3.1-nemotron-70b-instruct").
        NVIDIA_API_KEY : (str, optional)
            The API key for accessing NVIDIA's NIM models.
            (default: "").
        ENDPOINT_URL : (str, optional)
            The URL hosting your model, in case you are not using
            the public NIM.
            (default: "https://integrate.api.nvidia.com/v1").
    """
    def __init__(
        self,
        NVIDIA_NIM_MODEL: Optional[
            str] = "nvidia/llama-3.1-nemotron-70b-instruct",
        NVIDIA_API_KEY: Optional[str] = "",
        ENDPOINT_URL: Optional[str] = "https://integrate.api.nvidia.com/v1",
    ) -> None:
        self.NVIDIA_API_KEY = NVIDIA_API_KEY
        self.NIM_MODEL = NVIDIA_NIM_MODEL
        self.ENDPOINT_URL = ENDPOINT_URL

    def _process_score(self, response: str) -> float:
        """Uses 3 and 1 even though prompt says only 0, 2, 4.
        This is because LLMs don't always follow instructions.
        Credit to Gilberto.
        """
        for i in [4, 3, 2, 1, 0]:
            if str(i) in response:
                return i / 4
        return float("nan")

    def _average_scores(self, score0: float, score1: float):
        """Take the average of score0 and score1.
        Sometimes the LLM fail to respond or have no score in the response.
        In those cases the failed score is discarded.
        Credit to Gilberto.

        Args:
         score0 (float): judge accuracy score.
         score1 (float): judge accuracy score by permuting agent answer and
         ground truth.

        Returns:
            (float) average of score0 and score1 of both contains scores,
            otherwise pick the max.
        """
        score = float("nan")
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        else:
            score = max(score0, score1)
        return score

    def score(
        self,
        question: str,
        model_pred: str,
        correct_answer: str,
    ) -> float:
        """Args:
            question (str): The original question asked to the model.
            model_pred (str): The prediction made by the model.
            correct_answer (str): The actual correct answer to the question.

        Returns:
            score (float): score of 0-1, may be nan due to LLM judge failure.
                Evals should skip nan's when aggregating score.
        """
        prompt1 = SYSTEM_PROMPT_1.format(question=question,
                                         model_pred=model_pred,
                                         correct_answer=correct_answer)
        prompt2 = SYSTEM_PROMPT_2.format(question=question,
                                         model_pred=model_pred,
                                         correct_answer=correct_answer)
        score1 = float("nan")
        score2 = float("nan")
        for _retry in range(200):
            try:
                score1 = self._process_score(
                    call_NIM(prompt1, self.NVIDIA_API_KEY, self.NIM_MODEL,
                             self.ENDPOINT_URL, post_text=""))
                if not isnan(score1):
                    break
            except ImportError:
                raise
            except:  # noqa
                pass
        for _retry in range(20):
            try:
                score2 = self._process_score(
                    call_NIM(prompt2, self.NVIDIA_API_KEY, self.NIM_MODEL,
                             self.ENDPOINT_URL, post_text=""))
                if not isnan(score2):
                    break
            except ImportError:
                raise
            except:  # noqa
                pass

        return self._average_scores(score1, score2)
