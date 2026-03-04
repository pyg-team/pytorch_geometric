import os
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp

CLIENT_INITD = False

CLIENT = None
GLOBAL_NIM_KEY = ""
SYSTEM_PROMPT = "Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Separate each with a new line. Do not output anything else. Try to focus on key triples that form a connected graph."  # noqa
MAX_OUTER_RETRIES = 5  # Maximum number of times the entire multiprocessing job is retried. # noqa
RETRY_DELAY = 5  # Fixed sleep time (in seconds) between outer retries.
MAX_NIM_RETRIES = 200  # Maximum number of attempts to call the NIM API inside one worker.  # noqa
BASE_DELAY = 0.5  # Initial wait time before retrying a failed network call.


class TXT2KG():
    """A class to convert text data into a Knowledge Graph (KG) format.
    Uses NVIDIA NIMs + Prompt engineering by default.
    Default model `nvidia/llama-3.1-nemotron-70b-instruct`
    is on par or better than GPT4o in benchmarks.
    We need a high quality model to ensure high quality KG.
    Otherwise we have garbage in garbage out for the rest of the
    GNN+LLM RAG pipeline.

    Use local_lm flag for local debugging/dev. You still need to be able to
    inference a 14B param LLM, 'VAGOsolutions/SauerkrautLM-v2-14b-DPO'.
    Smaller LLMs did not work at all in testing.
    Note this 14B model requires a considerable amount of GPU memory.
    See examples/llm/txt2kg_rag.py for an example.

    Args:
        NVIDIA_NIM_MODEL : str, optional
            The name of the NVIDIA NIM model to use.
            (default: "nvidia/llama-3.1-nemotron-70b-instruct").
        NVIDIA_API_KEY : str, optional
            The API key for accessing NVIDIA's NIM models (default: "").
        ENDPOINT_URL : str, optional
            The URL hosting your model, in case you are not using
            the public NIM.
            (default: "https://integrate.api.nvidia.com/v1").
        local_LM : bool, optional
            A flag indicating whether a local Language Model (LM)
            should be used. This uses HuggingFace and will be slower
            than deploying your own private NIM endpoint. This flag
            is mainly recommended for dev/debug.
            (default: False).
        chunk_size : int, optional
            The size of the chunks in which the text data is processed
            (default: 512).
    """
    def __init__(
        self,
        NVIDIA_NIM_MODEL: Optional[
            str] = "nvidia/llama-3.1-nemotron-70b-instruct",
        NVIDIA_API_KEY: Optional[str] = "",
        ENDPOINT_URL: Optional[str] = "https://integrate.api.nvidia.com/v1",
        local_LM: bool = False,
        chunk_size: int = 512,
    ) -> None:
        self.local_LM = local_LM
        # Initialize the local LM flag and the NIM model info accordingly
        if self.local_LM:
            # If using a local LM, set the initd_LM flag to False
            self.initd_LM = False
        else:
            # If not using a local LM, store the provided NIM model info
            self.NVIDIA_API_KEY = NVIDIA_API_KEY
            self.NIM_MODEL = NVIDIA_NIM_MODEL
            self.ENDPOINT_URL = ENDPOINT_URL

        # Set the chunk size for processing text data
        self.chunk_size = chunk_size

        # Initialize counters and storage for parsing results
        self.doc_id_counter = 0
        self.relevant_triples = {}
        self.total_chars_parsed = 0
        self.time_to_parse = 0.0

    def save_kg(self, path: str) -> None:
        """Saves the relevant triples in the knowledge graph (KG) to a file.

        Args:
            path (str): The file path where the KG will be saved.

        Returns:
            None
        """
        torch.save(self.relevant_triples, path)

    def _chunk_to_triples_str_local(self, txt: str) -> str:
        # call LLM on text
        chunk_start_time = time.time()
        if not self.initd_LM:
            from torch_geometric.nn.nlp import LLM
            LM_name = "VAGOsolutions/SauerkrautLM-v2-14b-DPO"
            self.model = LLM(LM_name).eval()
            self.initd_LM = True
        out_str = self.model.inference(question=[txt + '\n' + SYSTEM_PROMPT],
                                       max_tokens=self.chunk_size)[0]
        # for debug
        self.total_chars_parsed += len(txt)
        self.time_to_parse += round(time.time() - chunk_start_time, 2)
        self.avg_chars_parsed_per_sec = self.total_chars_parsed / self.time_to_parse  # noqa
        return out_str

    def add_doc_2_KG(
        self,
        txt: str,
        QA_pair: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Add a document to the Knowledge Graph (KG).

        Args:
            txt (str): The text to extract triples from.
            QA_pair (Tuple[str, str]], optional):
                A QA pair to associate with the extracted triples.
                Useful for downstream evaluation.

        Returns:
        - None
        """
        if not self.local_LM:
            # Ensure NVIDIA_API_KEY is set before proceeding
            assert self.NVIDIA_API_KEY != '', \
                "Please init TXT2KG w/ NVIDIA_API_KEY or set local_lm=True"
        if QA_pair:
            # QA_pairs should be unique keys, check if already exists in KG
            if QA_pair in self.relevant_triples.keys():
                print("Warning: QA_Pair was already added to the set")
                print("Q=", QA_pair[0])
                print("A=", QA_pair[1])
                print("Previously parsed triples=",
                      self.relevant_triples[QA_pair])
                print("Skipping...")
            key = QA_pair
        else:
            # If no QA_pair, use the current doc_id_counter as the key
            key = self.doc_id_counter

        self.relevant_triples[key] = self._extract_relevant_triples(txt)

        # Increment the doc_id_counter for the next document
        self.doc_id_counter += 1

    def _extract_relevant_triples(
            self, txt: str, max_retries: int = MAX_OUTER_RETRIES,
            retry_delay: float = RETRY_DELAY) -> List[Tuple[str, str, str]]:
        # Handle empty text (context-less QA pairs)
        if txt == "":
            return []

        # Chunk the text into smaller pieces for processing
        chunks = _chunk_text(txt, chunk_size=self.chunk_size)

        if self.local_LM:
            # For debugging purposes...
            # process chunks sequentially on the local LM
            return _llm_then_python_parse(chunks, _parse_n_check_triples,
                                          self._chunk_to_triples_str_local)

        # Create deterministic chunk assignment
        import math
        num_procs = min(len(chunks), _get_num_procs())
        chunk_size = math.ceil(len(chunks) / num_procs)
        in_chunks_per_proc = [
            chunks[j * chunk_size:min((j + 1) * chunk_size, len(chunks))]
            for j in range(num_procs)
        ]

        # Run workers via starmap for deterministic ordering
        worker_args = [(
            rank,
            in_chunks_per_proc[rank],
            _parse_n_check_triples,
            _chunk_to_triples_str_cloud,
            self.NVIDIA_API_KEY,
            self.NIM_MODEL,
            self.ENDPOINT_URL,
        ) for rank in range(num_procs)]

        for attempt in range(max_retries):
            try:
                with mp.get_context("spawn").Pool(num_procs) as pool:
                    results = pool.starmap(_multiproc_helper, worker_args)
                break  # success

            except Exception as e:
                if attempt == max_retries - 1:
                    raise  # re-raise on final failure

                print(f"[Retry {attempt+1}/{max_retries}] "
                      f"Multiprocessing failed: {e}")
            time.sleep(retry_delay)

        return _merge_triples_deterministically(results)


known_reasoners = [
    "llama-3.1-nemotron-ultra-253b-v1",
    "kimi-k2-instruct",
    "nemotron-super-49b-v1_5",
    "gpt-oss",
]


def _chunk_to_triples_str_cloud(
        txt: str, GLOBAL_NIM_KEY='',
        NIM_MODEL="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        ENDPOINT_URL="https://integrate.api.nvidia.com/v1",
        post_text=SYSTEM_PROMPT) -> str:
    global CLIENT_INITD
    if not CLIENT_INITD:
        # We use NIMs since most PyG users may not be able to run a 70B+ model
        try:
            from openai import OpenAI
        except ImportError:
            quit(
                "Failed to import `openai` package, please install it and rerun the script"  # noqa
            )
        global CLIENT
        CLIENT = OpenAI(base_url=ENDPOINT_URL, api_key=GLOBAL_NIM_KEY)
        CLIENT_INITD = True
    txt_input = txt
    if post_text != "":
        txt_input += '\n' + post_text
    messages = []
    if any([model_name_str in NIM_MODEL
            for model_name_str in known_reasoners]):
        messages.append({"role": "system", "content": "detailed thinking on"})
    messages.append({"role": "user", "content": txt_input})
    completion = CLIENT.chat.completions.create(model=NIM_MODEL,
                                                messages=messages,
                                                temperature=0, top_p=1,
                                                max_tokens=1024, stream=True)
    out_str = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            out_str += chunk.choices[0].delta.content
    return out_str


def _parse_n_check_triples(triples_str: str) -> List[Tuple[str, str, str]]:
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
            split_trip = [(i[:-1].lower() if i[-1] == " " else i)
                          for i in split_trip]
            potential_trip = tuple(split_trip)
        except:  # noqa
            continue
        if 'tuple' in str(type(potential_trip)) and len(
                potential_trip
        ) == 3 and "note:" not in potential_trip[0].lower():
            # additional check for empty node/edge attrs
            if potential_trip[0] != '' and potential_trip[
                    1] != '' and potential_trip[2] != '':
                processed.append(potential_trip)
    return processed


def _llm_then_python_parse(chunks, py_fn, llm_fn, **kwargs):
    relevant_triples = []
    for chunk in chunks:
        relevant_triples += py_fn(llm_fn(chunk, **kwargs))
    return relevant_triples


def _multiproc_helper(rank, chunks_for_rank, py_fn, llm_fn, NIM_KEY, NIM_MODEL,
                      ENDPOINT_URL, max_retries=MAX_NIM_RETRIES,
                      base_delay=BASE_DELAY):

    for attempt in range(max_retries):
        try:
            return _llm_then_python_parse(
                chunks_for_rank,
                py_fn,
                llm_fn,
                GLOBAL_NIM_KEY=NIM_KEY,
                NIM_MODEL=NIM_MODEL,
                ENDPOINT_URL=ENDPOINT_URL,
            )

        except Exception:
            # Optional: restrict to network-related exceptions only
            if attempt == max_retries - 1:
                raise

            # exponential backoff with jitter
            from random import uniform
            sleep_time = base_delay * (2**min(attempt, 6))
            sleep_time += uniform(0, 0.1)
            time.sleep(sleep_time)


def _get_num_procs():
    if hasattr(os, "sched_getaffinity"):
        try:
            num_proc = len(os.sched_getaffinity(0)) / (2)
        except Exception:
            pass
    if num_proc is None:
        num_proc = os.cpu_count() / (2)
    return int(num_proc)


def _chunk_text(text: str, chunk_size: int = 512) -> list[str]:
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


Triple = Union[List[str], Tuple[str, ...]]

def _merge_triples_deterministically(
        triples: List[List[Triple]]
) -> List[Tuple[str, ...]]:
    """
    Flatten a list of lists of triples and return a deterministic,
    reproducible sorted list of tuples.

    Args:
        triples (List[List[Triple]]): A list of lists of triples, where each
            triple is a list or tuple of strings or other comparable values.
            Typically, each inner list comes from a worker.

    Returns:
        List[Tuple[str, ...]]: A flattened list of triples as tuples, sorted
            deterministically. Sorting is Unicode-safe and reproducible across
            Python versions using `str.casefold()`. Tuples are immutable to
            ensure hashability and stability in dicts/sets.
    """
    # Flatten all sublists and convert inner lists to tuples
    flat_triples = [
        tuple(t)
        for sublist in triples
        for t in sublist
    ]

    # Deterministic sort (Unicode-safe, casefold for strings)
    flat_triples.sort(
        key=lambda triple: tuple(
            s.casefold() if isinstance(s, str) else s
            for s in triple
        )
    )

    return flat_triples
