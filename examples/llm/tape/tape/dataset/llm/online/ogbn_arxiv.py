from typing import Dict

from tape.config import DatasetName
from tape.dataset.llm.engine import LlmOnlineEngineArgs, LlmResponseModel
from tape.dataset.llm.online.base import LlmOnlineEngine


class LlmOgbnArxivResponses(LlmOnlineEngine):
    def __init__(self, args: LlmOnlineEngineArgs,
                 class_id_to_label: Dict) -> None:
        super().__init__(args=args, dataset_name=DatasetName.OGBN_ARXIV)
        self.class_id_to_label = class_id_to_label
        self.system_message = ('Which arXiv CS sub-category does this '
                               'paper belong to?')

    def get_response_model(self) -> LlmResponseModel:
        topk = 5
        class_labels = {
            v['category'].replace('-', ' ').replace(',', ''): v['label']
            for v in self.class_id_to_label.values()
        }
        labels_list = list(class_labels.values())
        kwargs = dict(
            class_labels=class_labels, label_description=(
                f'Provide {topk} likely arXiv CS sub-categories ordered '
                'from most to least likely.'),
            label_examples=[labels_list[:topk], labels_list[topk:topk * 2]],
            reason_examples=[
                ('The paper is about a new dataset for scene text '
                 'detection and recognition, which is a topic '
                 'related to computer vision (cs.CV). '
                 'The paper also mentions the use of deep learning '
                 'techniques such as DeconvNet, which falls under '
                 'the sub-category of artificial intelligence (cs.AI).'
                 ' The dataset is annotated and involves text '
                 'recognition, which could also fall under the '
                 'sub-categories of information retrieval (cs.IR) and'
                 ' natural language processing (cs.CL). Finally, the '
                 'paper discusses the effectiveness of different '
                 'solutions, which could be evaluated using machine '
                 'learning techniques, falling under the sub-category'
                 ' of machine learning (cs.LG).'),
            ])
        return self.load_response_model_from_template(**kwargs)
