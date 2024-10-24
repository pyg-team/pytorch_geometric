import math
from typing import List, Optional, Tuple


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
        self.triples_per_doc_id = {}
        # keep track of which doc each triple comes from
        # useful for approximating recall of subgraph retrieval algos
        self.doc_id_counter = 0
        self.relevant_docs_per_q_a_pair = {}

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
        print("triples_str=", triples_str)
        print("triples_str split w backn=", triples_str.split("\n"))
        # (TODO) make pythonic logic to parse into triples

    def add_doc_2_KG(
        self,
        txt: str,
        QA_pair: Optional[Tuple[str, str]],
    ) -> None:
        # if QA_pair is not None, store with matching doc ids
        # useful for approximating recall
        chunks = [
            txt[i:min((i + 1) * self.chunk_size, len(txt))]
            for i in range(math.ceil(len(txt) / self.chunk_size))
        ]
        self.triples_per_doc_id[self.doc_id_counter] = []
        for chunk in chunks:
            self.triples_per_doc_id[
                self.doc_id_counter] += self.parse_n_check_triples(
                    self.chunk_to_triples_str(chunk))
        if QA_pair:
            if QA_pair in self.relevant_docs_per_q_a_pair.keys():
                self.relevant_docs_per_q_a_pair[QA_pair] += [
                    self.doc_id_counter
                ]
            else:
                self.relevant_docs_per_q_a_pair[QA_pair] = [
                    self.doc_id_counter
                ]
        self.doc_id_counter += 1
