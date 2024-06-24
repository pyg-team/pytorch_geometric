from typing import Dict

from tape.dataset.llm.engine import LlmOfflineEngineArgs
from tape.dataset.llm.offline.base import LlmOfflineEngine


class LlmOgbnArxivResponses(LlmOfflineEngine):
    def __init__(self, args: LlmOfflineEngineArgs,
                 class_id_to_label: Dict) -> None:
        super().__init__(args)
        self.class_id_to_label = class_id_to_label

    def get_system_prompt(self) -> str:
        topk = 5
        categories = []
        for v in self.class_id_to_label.values():
            category = v['category'].replace('-', ' ').replace(',', '')
            categories.append(f"{v['label']} // {category}")
        kwargs = dict(
            role="You're an experienced computer scientist.",
            categories=categories,
            label_description=(
                f'Contains {topk} arXiv CS sub-categories ordered '
                'from most to least likely.', ),
        )
        prompt = self.load_system_prompt_from_template(**kwargs)
        return prompt
