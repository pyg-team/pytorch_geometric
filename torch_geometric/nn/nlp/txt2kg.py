from enum import Enum
from typing import List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class TXT2KG():
    # Uses NVIDIA NIMs + Prompt engineering to extract KG from text
    def __init__(
        self,
        NVIDIA_API_KEY,
        chunk_size=512,
    ) -> None:
        # We use NIMs since most PyG users may not be able to run a 70B+ model
        from openai import OpenAI

        self.client = OpenAI(
          base_url = "https://integrate.api.nvidia.com/v1",
          api_key = NVIDIA_API_KEY
        )
        self.chunk_size = 512
        self.system_prompt = "Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Seperate each with a new line. Do not output anything else.”"
        self.model = "nvidia/llama-3.1-nemotron-70b-instruct"
        self.triples = []



    def parse_txt_2_KG(self, txt: str) -> None:
        chunks = [txt[i:(i+1) * self.chunk_size] for i in range(math.ceil(len(txt)/self.chunk_size))]
        for chunk in chunks:
            self.triples += parse_n_check_triples(chunk_to_triples_str(chunk))

    def chunk_to_triples_str(self, txt: str) -> List[Tuple[str, str, str]]:
        # call LLM on text
        completion = self.client.chat.completions.create(
          model=self.model,
          messages=[{"role":"user","content":txt + '\n' + self.system_prompt}],
          temperature=0,
          top_p=1,
          max_tokens=1024,
          stream=True
        )
        out_str = ""
        for chunk in completion:
          if chunk.choices[0].delta.content is not None:
            out_str += chunk.choices[0].delta.content
        return out_str

    def parse_n_check_triples(self, triples: str) -> List[Tuple[str, str, str]]:
        # use pythonic checks for triples

    def get_KG(self, ) -> List[Tuple[str, str, str]]:
        return self.triples

