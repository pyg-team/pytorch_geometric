import random
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

import instructor
from litellm import completion
from tape.config import DatasetName
from tape.dataset.llm.engine import (
    LlmEngine,
    LlmOnlineEngineArgs,
    LlmResponseModel,
)
from tape.dataset.llm.online.cache import llm_responses_cache, setup_cache
from tape.dataset.parser import Article
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from torch_geometric.template import module_from_template


class LlmOnlineEngine(LlmEngine):
    def __init__(self, args: LlmOnlineEngineArgs,
                 dataset_name: DatasetName) -> None:
        super().__init__(args)
        self.dataset_name = dataset_name.value
        self.client = instructor.from_litellm(completion)
        setup_cache(cache_dir=Path(args.cache_dir) /
                    f'tape_llm_responses/{dataset_name.value}')
        self._response_model = None

    @abstractmethod
    def get_response_model(self) -> LlmResponseModel:
        pass

    @property
    def response_model(self) -> LlmResponseModel:
        if self._response_model is None:
            self._response_model = self.get_response_model()
        return self._response_model

    def __call__(self, article: Article) -> Optional[LlmResponseModel]:
        messages = [
            dict(role='system', content=self.system_message),
            dict(
                role='user', content='Title: {}\nAbstract: {}'.format(
                    article.title, article.abstract))
        ]
        response = None
        rpm = self.args.rate_limit_per_minute
        try:
            response = self._completion_with_backoff(
                model=self.args.model,
                messages=messages,
                response_model=self.response_model,
                delay=60.0 / rpm if rpm else
                0,  # Adding delay to a request to avoid hitting the rate limit
                **self.args.sampling_kwargs)
        except Exception as e:
            print('Max retries reached. Failed to get a successful response. '
                  f'Error: {e}')

        return response

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(6))
    @llm_responses_cache
    def _completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create_with_completion(**kwargs)

    def get_responses_from_articles(
            self, articles: List[Article]) -> List[LlmResponseModel]:
        responses = []
        for article in tqdm(articles, total=len(articles),
                            desc='Fetching LLM responses'):
            if not (response := self(article)):
                raise ValueError('LLM response cannot be empty!')
            response.label = response.label.value  # Convert Enum to str
            responses.append(response)
        return responses

    def load_response_model_from_template(self, **kwargs) -> LlmResponseModel:
        uid = '%06x' % random.randrange(16**6)
        path = Path(__file__, '..').resolve()
        module = module_from_template(
            module_name=f'response_model-{self.dataset_name}-{uid}',
            template_path=path / 'response_model.jinja',
            tmp_dirname='response_model', **kwargs)
        return module.Classification
