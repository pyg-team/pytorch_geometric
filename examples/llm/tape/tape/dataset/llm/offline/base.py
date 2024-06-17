import json
import os
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

from jinja2 import Environment, FileSystemLoader
from tape.dataset.llm.engine import (
    LlmEngine,
    LlmOfflineEngineArgs,
    LlmResponseModel,
)
from tape.dataset.parser import Article
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LlmOfflineEngine(LlmEngine):
    def __init__(self, args: LlmOfflineEngineArgs):
        super().__init__(args)
        # Update `huggingface_hub` default cache dir
        os.environ['HF_HOME'] = args.cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model, cache_dir=args.cache_dir)
        self.llm = LLM(model=args.model, **(args.engine_kwargs or {}))
        self.sampling_params = SamplingParams(**(args.sampling_kwargs or {}))
        self._system_prompt = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self.get_system_prompt()
        return self._system_prompt

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    def _prepare_conversation(
            self, articles: List[Article]) -> List[List[Dict[str, str]]]:
        conversation = []
        for article in articles:
            prompt = f'Title: {article.title}\nAbstract: {article.abstract}'
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ]
            conversation.append(messages)
        return conversation

    def __call__(
        self,
        articles: Union[Article, List[Article]],
        return_prompt: bool = False,
        strict: bool = False,  # Whether to use strict json parsing
        max_retries: int = 3,
    ) -> Union[Dict[str, Union[str, Dict]], List[Dict[str, Union[str, Dict]]]]:

        single_article = isinstance(articles, Article)
        if single_article:
            articles = [articles]
        conversation = self._prepare_conversation(articles)

        apply_chat_template = partial(self.tokenizer.apply_chat_template,
                                      add_generation_prompt=True,
                                      tokenize=not return_prompt)
        kwargs = dict(use_tqdm=False, sampling_params=self.sampling_params)
        prompt_key = 'prompts' if return_prompt else 'prompt_token_ids'
        kwargs[prompt_key] = apply_chat_template(conversation)

        outputs = self.llm.generate(**kwargs)
        max_retries = self.args.max_retries or max_retries
        for i in range(len(outputs)):
            output = outputs[i].outputs[0].text
            if strict:
                retries = 0
                json_output = None
                while retries < max_retries:
                    try:
                        json_output = json.loads(output)
                        break
                    except Exception:
                        retries += 1
                        print(f'Retry {retries}/{max_retries} '
                              'after exception: {e}')
                        kwargs[prompt_key] = apply_chat_template(
                            conversation[i:i + 1])
                        output = self.llm.generate(**kwargs)[0].outputs[0].text
                output = json_output
            outputs[i] = dict(input=outputs[i].prompt, output=output)

        return outputs[0] if single_article else outputs

    def get_responses_from_articles(
            self, articles: List[Article]) -> List[LlmResponseModel]:
        responses = [None] * len(articles)
        batch_size = self.args.batch_size
        for start in tqdm(range(0, len(articles), batch_size),
                          total=len(articles) // batch_size):
            results = self(articles[start:start + batch_size], strict=True)
            for idx, result in zip(range(start, start + batch_size), results):
                if result and result['output']:
                    responses[idx] = LlmResponseModel(**result['output'])
        return responses

    def load_system_prompt_from_template(self, **kwargs) -> str:
        file_loader = FileSystemLoader(Path(__file__, '..').resolve())
        env = Environment(loader=file_loader)
        template = env.get_template('prompt.jinja')
        prompt = template.render(**kwargs)
        return prompt
