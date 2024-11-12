import math
import time
from typing import List, Optional, Tuple

import torch


class TXT2KG():
    """Uses NVIDIA NIMs + Prompt engineering to extract KG from text
    nvidia/llama-3.1-nemotron-70b-instruct is on par or better than GPT4o
    in benchmarks. We need a high quality model to ensure high quality KG.
    Otherwise garbage in garbage out.
    Use local_lm flag for debugging. You still need to be able to inference
    a 14B param LLM, 'VAGOsolutions/SauerkrautLM-v2-14b-DPO'. Smaller LLMs
    did not work at all in testing.
    Note this 14B model requires a considerable amount of VRAM.
    """
    def __init__(
        self,
        NVIDIA_API_KEY: Optional[str] = '',
        local_LM: bool = False,
        chunk_size: int = 512,
    ) -> None:
        self.local_LM = local_LM
        if self.local_LM:
            self.initd_LM = False
        else:
            # We use NIMs since most PyG users may not be able to run a 70B+ model
            assert NVIDIA_API_KEY != '', "Please pass NVIDIA_API_KEY or set local_small_lm flag to True"
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=NVIDIA_API_KEY)
            self.model = "nvidia/llama-3.1-nemotron-70b-instruct"

        self.chunk_size = 512
        self.system_prompt = "Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Seperate each with a new line. Do not output anything else.”"
        # useful for approximating recall of subgraph retrieval algos
        self.doc_id_counter = 0
        self.relevant_triples = {}
        self.total_chars_parsed = 0
        self.time_to_parse = 0.0

    def save_kg(self, path: str) -> None:
        torch.save(self.relevant_triples, path)

    def chunk_to_triples_str(self, txt: str) -> str:
        # call LLM on text
        chunk_start_time = time.time()
        if self.local_LM:
            if not self.initd_LM:
                from torch_geometric.nn.nlp import LLM
                LM_name = "VAGOsolutions/SauerkrautLM-v2-14b-DPO"
                self.model = torch.compile(LLM(LM_name, num_params=14).eval())
                self.initd_LM = True
            out_str = self.model.inference(
                question=[txt + '\n' + self.system_prompt],
                max_tokens=self.chunk_size)[0]
        else:
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
        self.total_chars_parsed += len(txt)
        self.time_to_parse += round(time.time() - chunk_start_time, 2)
        self.avg_chars_parsed_per_sec = self.total_chars_parsed / self.time_to_parse
        return out_str

    def parse_n_check_triples(self,
                              triples_str: str) -> List[Tuple[str, str, str]]:
        # use pythonic checks for triples
        processed = []
        split_by_newline = triples_str.split("\n")
        # sometimes LLM fails to obey the prompt
        if len(split_by_newline) != 1:
            split_triples = split_by_newline
            llm_obeyed = True
        else:
            # handles form "(e, r, e) (e, r, e) ...""
            split_triples = triples_str[1:].split(") (")
            llm_obeyed = False
        for triple_str in split_triples:
            try:
                if llm_obeyed:
                    # remove parenthesis for parsing
                    triple_str = triple_str[1:-1]
                potential_trip = tuple(triples_str.split(','))
                print("potential_trip =", potential_trip)
            except:  # noqa
                print("Failed to parse triple =", triple_str)
                print("Full str =", triples_str)
                continue
            if 'tuple' in str(
                    type(potential_trip)) and len(potential_trip) == 3:
                processed.append(potential_trip)
        print("processed=", processed)
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
            assert QA_pair not in self.relevant_triples.keys()
            key = QA_pair
        else:
            key = self.doc_id_counter
        self.relevant_triples[key] = []
        for chunk in chunks:
            self.relevant_triples[key] += self.parse_n_check_triples(
                self.chunk_to_triples_str(chunk))
        self.doc_id_counter += 1
