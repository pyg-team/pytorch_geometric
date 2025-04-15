import os
import time
from typing import List, Optional, Tuple

import torch
import torch.multiprocessing as mp

CLIENT_INITD = False

CLIENT = None
GLOBAL_NIM_KEY = ""
SYSTEM_PROMPT = "Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Seperate each with a new line. Do not output anything else.”"


class TXT2KG():
    """Uses NVIDIA NIMs + Prompt engineering to extract KG from text.
    `nvidia/llama-3.1-nemotron-70b-instruct` is on par or better than GPT4o
    in benchmarks. We need a high quality model to ensure high quality KG.
    Otherwise we have garbage in garbage out for the rest of the
    GNN+LLM RAG pipeline.

    Use local_lm flag for local debugging/dev. You still need to be able to
    inference a 14B param LLM, 'VAGOsolutions/SauerkrautLM-v2-14b-DPO'.
    Smaller LLMs did not work at all in testing.
    Note this 14B model requires a considerable amount of VRAM.
    """
    def __init__(
        self,
        NVIDIA_NIM_MODEL: Optional[
            str] = "nvidia/llama-3.1-nemotron-70b-instruct",
        NVIDIA_API_KEY: Optional[str] = "",
        local_LM: bool = False,
        chunk_size: int = 512,
    ) -> None:
        self.local_LM = local_LM
        if self.local_LM:
            self.initd_LM = False
        else:
            assert NVIDIA_API_KEY != '', "Please pass NVIDIA_API_KEY or set local_small_lm flag to True"
            self.NVIDIA_API_KEY = NVIDIA_API_KEY
            self.NIM_MODEL = NVIDIA_NIM_MODEL

        self.chunk_size = 512
        # useful for approximating recall of subgraph retrieval algos
        self.doc_id_counter = 0
        self.relevant_triples = {}
        self.total_chars_parsed = 0
        self.time_to_parse = 0.0

    def save_kg(self, path: str) -> None:
        torch.save(self.relevant_triples, path)

    def chunk_to_triples_str_local(self, txt: str) -> str:
        # call LLM on text
        chunk_start_time = time.time()
        if not self.initd_LM:
            from torch_geometric.nn.nlp import LLM
            LM_name = "VAGOsolutions/SauerkrautLM-v2-14b-DPO"
            self.model = LLM(LM_name, num_params=14).eval()
            self.initd_LM = True
        out_str = self.model.inference(question=[txt + '\n' + SYSTEM_PROMPT],
                                       max_tokens=self.chunk_size)[0]
        # for debug
        self.total_chars_parsed += len(txt)
        self.time_to_parse += round(time.time() - chunk_start_time, 2)
        self.avg_chars_parsed_per_sec = self.total_chars_parsed / self.time_to_parse
        return out_str

    def add_doc_2_KG(
        self,
        txt: str,
        QA_pair: Optional[Tuple[str, str]],
    ) -> None:
        chunks = chunk_text(txt, chunk_size=self.chunk_size)
        if QA_pair:
            # QA_pairs should be unique keys
            assert QA_pair not in self.relevant_triples.keys()
            key = QA_pair
        else:
            key = self.doc_id_counter
        if self.local_LM:
            # just for debug, no need to scale
            self.relevant_triples[key] = llm_then_python_parse(
                chunks, parse_n_check_triples, self.chunk_to_triples_str_local)
        else:
            num_procs = min(len(chunks), get_num_procs())
            meta_chunk_size = int(len(chunks) / num_procs)
            in_chunks_per_proc = {
                j:
                chunks[j * meta_chunk_size:min((j + 1) *
                                               meta_chunk_size, len(chunks))]
                for j in range(num_procs)
            }
            mp.spawn(
                multiproc_helper,
                args=(in_chunks_per_proc, parse_n_check_triples,
                      chunk_to_triples_str_cloud, self.NVIDIA_API_KEY,
                      self.NIM_MODEL), nprocs=num_procs)
            self.relevant_triples[key] = []
            for rank in range(num_procs):
                self.relevant_triples[key] += torch.load(
                    "/tmp/outs_for_proc_" + str(rank))
                os.remove("/tmp/outs_for_proc_" + str(rank))
        self.doc_id_counter += 1


def chunk_to_triples_str_cloud(
        txt: str, GLOBAL_NIM_KEY='',
        NIM_MODEL="nvidia/llama-3.1-nemotron-70b-instruct") -> str:
    global CLIENT_INITD
    if not CLIENT_INITD:
        # We use NIMs since most PyG users may not be able to run a 70B+ model
        from openai import OpenAI
        global CLIENT
        CLIENT = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                        api_key=GLOBAL_NIM_KEY)
        CLIENT_INITD = True
    completion = CLIENT.chat.completions.create(
        model=NIM_MODEL, messages=[{
            "role": "user",
            "content": txt + '\n' + SYSTEM_PROMPT
        }], temperature=0, top_p=1, max_tokens=1024, stream=True)
    out_str = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            out_str += chunk.choices[0].delta.content
    return out_str


def parse_n_check_triples(triples_str: str) -> List[Tuple[str, str, str]]:
    # use pythonic checks for triples
    processed = []
    split_by_newline = triples_str.split("\n")
    # sometimes LLM fails to obey the prompt
    if len(split_by_newline) > 1:
        split_triples = split_by_newline
        llm_obeyed = True
    else:
        # handles form "(e, r, e) (e, r, e) ... (e, r, e)""
        split_triples = triples_str[1:-1].split(") (")
        llm_obeyed = False
    for triple_str in split_triples:
        try:
            if llm_obeyed:
                # remove parenthesis and single quotes for parsing
                triple_str = triple_str.replace("(", "").replace(")",
                                                                 "").replace(
                                                                     "'", "")
            split_trip = triple_str.split(',')
            # remove blank space at beginning or end
            split_trip = [(i[1:] if i[0] == " " else i) for i in split_trip]
            split_trip = [(i[:-1] if i[-1] == " " else i) for i in split_trip]
            potential_trip = tuple(split_trip)
        except:  # noqa
            continue
        if 'tuple' in str(type(potential_trip)) and len(potential_trip) == 3:
            processed.append(potential_trip)
    return processed


def llm_then_python_parse(chunks, py_fn, llm_fn, **kwargs):
    relevant_triples = []
    for chunk in chunks:
        relevant_triples += py_fn(llm_fn(chunk, **kwargs))
    return relevant_triples


def multiproc_helper(rank, in_chunks_per_proc, py_fn, llm_fn, NIM_KEY,
                     NIM_MODEL):
    out = llm_then_python_parse(in_chunks_per_proc[rank], py_fn, llm_fn,
                                GLOBAL_NIM_KEY=NIM_KEY, NIM_MODEL=NIM_MODEL)
    torch.save(out, "/tmp/outs_for_proc_" + str(rank))


def get_num_procs():
    if hasattr(os, "sched_getaffinity"):
        try:
            num_proc = len(os.sched_getaffinity(0)) / (2)
        except Exception:
            pass
    if num_proc is None:
        num_proc = os.cpu_count() / (2)
    return int(num_proc)


def chunk_text(text: str, chunk_size: int = 512) -> list[str]:
    """Function to chunk text into sentence-based segments.
    Co-authored with Claude AI.
    """
    # If the input text is empty or None, return an empty list
    if not text:
        return []

    # List of punctuation marks that typically end sentences
    sentence_endings = '.!?'

    # List to store the resulting chunks
    chunks = []

    # Continue processing the entire text
    while text:
        # If the remaining text is shorter than chunk_size, add it and break
        if len(text) <= chunk_size:
            chunks.append(text.strip())
            break

        # Start with the maximum possible chunk
        chunk = text[:chunk_size]

        # Try to find the last sentence ending within the chunk
        best_split = chunk_size
        for ending in sentence_endings:
            # Find the last occurrence of the ending punctuation
            last_ending = chunk.rfind(ending)
            if last_ending != -1:
                # Ensure we include the punctuation and any following space
                best_split = min(
                    best_split, last_ending + 1 +
                    (1 if last_ending + 1 < len(chunk)
                     and chunk[last_ending + 1].isspace() else 0))

        # Adjust to ensure we don't break words
        # If the next character is a letter, find the last space
        if best_split < len(text) and text[best_split].isalpha():
            # Find the last space before the current split point
            space_split = text[:best_split].rfind(' ')
            if space_split != -1:
                best_split = space_split

        # Append the chunk, ensuring it's stripped
        chunks.append(text[:best_split].strip())

        # Remove the processed part from the text
        text = text[best_split:].lstrip()

    return chunks
