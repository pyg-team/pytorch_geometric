from typing import Dict

from tape.dataset.llm.engine import LlmOfflineEngineArgs
from tape.dataset.llm.offline.base import LlmOfflineEngine


class LlmPubmedResponses(LlmOfflineEngine):
    def __init__(self, args: LlmOfflineEngineArgs,
                 class_id_to_label: Dict) -> None:
        super().__init__(args)
        self.class_id_to_label = class_id_to_label

    def get_system_prompt(self) -> str:
        kwargs = dict(
            role="You're an experienced medical doctor.",
            categories=[v['label'] for v in self.class_id_to_label.values()],
            label_description=(
                'Contains the category (or categories if multiple options '
                'apply) ordered from most to least likely.'),
        )
        prompt = self.load_system_prompt_from_template(**kwargs)
        return prompt
