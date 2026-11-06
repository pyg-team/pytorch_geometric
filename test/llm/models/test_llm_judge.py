import numpy as np

from torch_geometric.llm.models import LLMJudge


def test_llm_judge():
    judge = LLMJudge()
    assert judge._process_score('1234') == 1.0
    assert judge._average_scores(1, 3) == 2
    assert judge._average_scores(-1, 3) == 3

    assert np.isnan(judge.score('question', 'model_pred', 'correct_answer'))
