import math
from typing import List, Optional, Tuple

import torch


class TXT2KG():
    """Uses NVIDIA NIMs + Prompt engineering to extract KG from text
    nvidia/llama-3.1-nemotron-70b-instruct is on par or better than GPT4o
    in benchmarks. We need a high quality model to ensure high quality KG.
    Otherwise garbage in garbage out.
    """
    def __init__(
        self,
        NVIDIA_API_KEY,
        chunk_size=512,
    ) -> None:
        # We use NIMs since most PyG users may not be able to run a 70B+ model
        from openai import OpenAI

        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                             api_key=NVIDIA_API_KEY)
        self.chunk_size = 512
        self.system_prompt = "Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Seperate each with a new line. Do not output anything else.”"
        self.model = "nvidia/llama-3.1-nemotron-70b-instruct"
        # useful for approximating recall of subgraph retrieval algos
        self.doc_id_counter = 0
        self.relevant_triples = {}

    def save_kg(self, path: str) -> None:
        torch.save(self.relevant_triples, path)

    def load_kg(self, path: str) -> None:
        self.relevant_triples = torch.load(path)

    def chunk_to_triples_str(self, txt: str) -> str:
        # call LLM on text
        completion = self.client.chat.completions.create(
            model=self.model, messages=[{
                "role":
                "user",
                "content":
                txt + '\n' + self.system_prompt
            }], temperature=0, top_p=1, max_tokens=1024, stream=True)
        out_str = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                out_str += chunk.choices[0].delta.content
        return out_str

    def parse_n_check_triples(self,
                              triples_str: str) -> List[Tuple[str, str, str]]:
        # use pythonic checks for triples
        processed = []
        for triple_str in triples_str.split("\n"):
            try:
                potential_trip = eval(triple_str)
            except:  # noqa
                continue
            if 'tuple' in str(type(potential_trip)):
                processed.append(potential_trip)
        return processed

    def add_doc_2_KG(
        self,
        txt: str,
        QA_pair: Optional[Tuple[str, str]],
    ) -> None:
        chunks = [
            txt[i * self.chunk_size:min((i + 1) * self.chunk_size, len(txt))]
            for i in range(math.ceil(len(txt) / self.chunk_size))
        ]
        if QA_pair:
            # QA_pairs should be unique keys
            assert QA_pair not in self.relevant_docs_per_q_a_pair.keys()
            key = QA_pair
        else:
            key = self.doc_id_counter
        self.relevant_triples[key] = []
        for chunk in chunks:
            self.relevant_triples[key] += self.parse_n_check_triples(
                self.chunk_to_triples_str(chunk))
        self.doc_id_counter += 1
