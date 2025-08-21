import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import List, Optional, Tuple

import torch
import torch.multiprocessing as mp

CLIENT_INITD = False

CLIENT = None
GLOBAL_NIM_KEY = ""


SYSTEM_PROMPT = """
You are a helpful assistant that converts text into a list of knowledge triples with the form ('entity', 'relation', 'entity').
Separate each with a new line. Do not output anything else.
Try to focus on key triples that form a connected graph.
"""

TRIPLES_SYS_PROMPT_NEW = (
    "You are an expert that can extract knowledge triples with the form `('entity', 'relation', 'entity)` "
    "from a text, mainly using entities from the entity list given by the user. Keep relations 2 words max."
    "Separate each with a new line. Do not output anything else (no notes, no explanations, etc)."
)


# defines the interface for chunk-wise actions (e.g. entity extraction, triple extraction, etc.)
class Action(ABC):
    # function that will parse the LLM output
    @abstractmethod
    def parse(self, raw_result: Any) -> Any:
        pass


# defines the interface for remote chunk-wise actions (e.g. entity extraction, triple extraction, etc.)
class RemoteAction(Action):
    # function that will prepare the prompt for the LLM
    @abstractmethod
    def prepare_prompt(self, chunk: str,
                       previous_results: Any = None) -> List[Dict[str, str]]:
        pass


# defines the interface for entity resolution
class EntityResolver(ABC):
    @abstractmethod
    def __call__(self, entities: List[str],
                 triples: List[Tuple[str, str, str]]) -> List[str]:
        pass


class LLMEntityExtractor(RemoteAction):
    def __init__(self):
        self.system_prompt = """
        Extract key entities from the given text. Extracted entities are nouns, verbs, or adjectives, particularly regarding sentiment. Keep them 2 words max. This is for an extraction task, please be thorough and accurate to the reference text.
        Do not output anything else.
        Output format:
        [
            "Entity1",
            "Entity2",
            "Entity3"
        ]
    """ # noqa

    def parse(self, raw_result: str) -> List[str]:
        entities = []

        cleaned_text = raw_result.strip()
        if (cleaned_text.startswith('[') and cleaned_text.endswith(']')):
            try:
                entities = json.loads(cleaned_text)
                entities = [str(entity) for entity in entities if entity]
                return entities
            except json.JSONDecodeError:
                pass
        # fallback when json fails
        lines = cleaned_text.replace('[', '').replace(']', '').split('\n')
        for line in lines:
            line = line.strip().strip('"\'').strip(',').strip()
            if line and not line.startswith(
                    '#') and not line.lower().startswith('entity'):
                entities.append(line)
        return entities

    def prepare_prompt(self, chunk: str,
                       previous_result: Any = None) -> List[Dict[str, str]]:
        messages = [{
            "role": "system",
            "content": self.system_prompt
        }, {
            "role": "user",
            "content": chunk
        }]
        return messages

    def __call__(self, chunks: List[str]) -> List[str]:
        return []


class EntityAugmenter(Action):
    def parse(self, out: str) -> List[str]:
        local_entities = out
        local_entities = [l_e.lower() for l_e in local_entities]
        local_entities = list(set(local_entities))
        # remove trailing spaces, commas, quotes, etc.
        local_entities = [
            l_e.strip().strip(',').strip('"').strip("'")
            for l_e in local_entities
        ]
        # filter out huge entities
        local_entities = [l_e for l_e in local_entities if len(l_e) < 64]
        return local_entities


class LLMTripleExtractor(RemoteAction):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def prepare_prompt(self, chunk: str,
                       previous_result: Any = None) -> List[Dict[str, str]]:
        entities = previous_result

        augmented_chunk = f"Entities: {entities}\nText: {chunk}"
        messages = [{
            "role": "system",
            "content": self.system_prompt
        }, {
            "role": "user",
            "content": augmented_chunk
        }]
        return messages

    def parse(self, raw_result: str) -> List[Tuple[str, str, str]]:
        triples_str = raw_result
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
                    triple_str = triple_str.replace("(", "").replace(
                        ")", "").replace("'", "")
                split_trip = triple_str.split(',')
                if len(split_trip) != 3:
                    # LLM sometimes output triples in the form `entity - relation - entity`
                    split_trip = triple_str.split('-')
                # remove blank space at beginning or end
                split_trip = [(i[1:] if i[0] == " " else i)
                              for i in split_trip]
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

    def __call__(self, chunks: List[str]) -> List[Tuple[str, str, str]]:
        return []


class RemoteLLMCaller:
    def __init__(self, model: str, api_key: str, endpoint_url: str,
                 max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        assert api_key != '', \
            "Please init TXT2KG w/ NVIDIA_API_KEY or set local_lm=True"
        self.max_tokens = max_tokens

    def __call__(self, messages: List[Dict[str, str]]) -> List[Any]:
        global CLIENT_INITD
        if not CLIENT_INITD:
            # We use NIMs since most PyG users may not be able to run a 70B+ model
            from openai import OpenAI
            global CLIENT
            CLIENT = OpenAI(base_url=self.endpoint_url, api_key=self.api_key)
            CLIENT_INITD = True
        completion = CLIENT.chat.completions.create(model=self.model,
                                                    messages=messages,
                                                    temperature=0, top_p=1,
                                                    max_tokens=self.max_tokens,
                                                    stream=True)
        out_str = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                out_str += chunk.choices[0].delta.content
        return out_str


class LLMEntityResolver(EntityResolver):
    def __init__(self, remote_llm_caller: RemoteLLMCaller):
        self.remote_llm_caller = remote_llm_caller
        self.system_prompt = """
You are given a list of entity strings.
1. Scan the list and, if possible, select exactly one cluster that meets BOTH of these criteria:
   - The entities point to the same real-world concept.
   - Their differences are only surface-level (e.g., tense, singular vs. plural, stemming, capitalization) they are near-synonyms with clearly overlapping meaning.
2. Output that cluster as a JSON array containing the original strings, in the order they appeared AND the string representing the cluster the best on the next line.
  [
  "using",
  "used",
  "utilizing"
  ]
  "uses"
  DO NOT output anything else.
3. If no clear cluster exists, output an empty JSON array: [] and nothing else.
AGAIN, DO NOT output anything else.
"""

        self.match_pattern = re.compile(r'(\[[^\]]*\])\s*("[^"]*")',
                                        re.VERBOSE | re.MULTILINE)

    def _process(self, raw_result: str) -> List[str]:
        try:
            matches = self.match_pattern.findall(raw_result)
            if len(matches) == 0:
                return [], ""

            # last match, in case of multiple matches (since some models are "reasoning", even tho they should not lol)
            item_list_str, summary_word = matches[-1]
            array = json.loads(item_list_str)
            array = [str(entity).strip(" '*-").lower() for entity in array]
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            return [], ""

        return array, summary_word

    def _prepare_prompt(self, items: List[str]) -> List[Dict[str, str]]:
        messages = [{
            "role": "system",
            "content": self.system_prompt
        }, {
            "role": "user",
            "content": str(items)
        }]
        return messages

    def _iterative_clustering(self, items: List[str],
                              num_iters: int = 5) -> List[str]:
        items_to_original_id = {item: i for i, item in enumerate(items)}

        # contains the mapping of the new items to the original items

        # map one item to a cluster of items
        cluster_to_items_mapping = {}

        print("before resolution: ", items_to_original_id.keys())

        for shots in range(num_iters):
            messages = self._prepare_prompt(items_to_original_id.keys())

            raw_result = self.remote_llm_caller(messages)
            print(f"Raw result: {raw_result}")
            try:
                ret_items, summary_word = self._process(raw_result)
            except Exception as e:
                print(
                    f"[_iterative_clustering] Failed to process LLM output: {type(e).__name__}: {str(e)}"
                )
                continue
            print(f"Summary word: {summary_word}")
            # handling diff kind of wrong LLM outputs
            if ret_items is None:
                continue
            if len(ret_items) != 0 and not summary_word:
                continue

            # handles hallucinated items
            ret_items = [
                item for item in ret_items if item in items_to_original_id
            ]
            # handles empty items
            ret_items = list({item for item in ret_items if item})
            print(
                f"Ret items (after filtering hallucinated items): {ret_items}")
            summary_word = summary_word.strip(" '*-").lower()
            cluster_to_items_mapping[summary_word] = [
                items_to_original_id[item] for item in ret_items
            ]

        return cluster_to_items_mapping

    def _iterative_resolution(self, items: List[str],
                              num_iters: int = 5) -> List[str]:
        items_to_original_id = {item: i for i, item in enumerate(items)}

        # contains the mapping of the new items to the original items
        items_to_id_mapping = {}

        new_items = []

        print("before resolution: ", items_to_original_id.keys())

        for shots in range(num_iters):
            messages = self._prepare_prompt(items_to_original_id.keys())
            raw_result = self.remote_llm_caller(messages)
            print(f"Raw result: {raw_result}")
            try:
                ret_items, summary_word = self._process(raw_result)
            except Exception as e:
                print(
                    f"[_iterative_resolution] Failed to process LLM output: {type(e).__name__}: {str(e)}"
                )
                continue
            print(f"Summary word: {summary_word}")
            # handling diff kind of wrong LLM outputs
            if ret_items is None:
                continue
            if len(ret_items) != 0 and not summary_word:
                continue

            # handles hallucinated items
            ret_items = [
                item for item in ret_items if item in items_to_original_id
            ]
            # handles empty items
            ret_items = list({item for item in ret_items if item})
            print(
                f"Ret items (after filtering hallucinated items): {ret_items}")

            summary_word = summary_word.strip(" '*-").lower()
            new_items.append(summary_word)

            for item in ret_items:
                items_to_id_mapping[
                    items_to_original_id[item]] = len(new_items) - 1
                items_to_original_id.pop(item)
            if len(items_to_original_id) == 0:
                break

        # clean up new_items one more time
        cleaned_new_items = []
        for item in new_items:
            match = re.search(r'"([^"]*)"', item)
            if match:
                cleaned_new_items.append(match.group(1))
            else:
                cleaned_new_items.append(item)
        new_items = cleaned_new_items

        # handles remaining items
        for item in items_to_original_id:
            items_to_id_mapping[items_to_original_id[item]] = len(new_items)
            new_items.append(item)

        print(f"Resolved items in {shots} shots")

        print(f"Items to id mapping: {new_items}")

        return new_items, items_to_id_mapping

    def _clean_and_deduplicate(
            self, items: List[str]) -> Tuple[List[str], Dict[int, int]]:
        cleaned_items = []
        item_mapping = {}
        for i, item in enumerate(items):
            cleaned_item = item.strip(" '*-").lower().replace("_", " ")
            if cleaned_item in cleaned_items:
                item_mapping[i] = cleaned_items.index(cleaned_item)
            else:
                item_mapping[i] = len(cleaned_items)
                cleaned_items.append(cleaned_item)
        return cleaned_items, item_mapping

    def __call__(
            self, triples: List[Tuple[str, str,
                                      str]]) -> List[Tuple[str, str, str]]:
        ents = {}
        rels = {}
        triples_by_ids = {}
        for i, (h, r, t) in enumerate(triples):
            if h not in ents:
                ents[h] = len(ents)
            if t not in ents:
                ents[t] = len(ents)
            if r not in rels:
                rels[r] = len(rels)
            triples_by_ids[i] = (ents[h], rels[r], ents[t])
        ents = list(ents)
        rels = list(rels)

        print(f"Entities: {ents}")
        print(f"Relations: {rels}")

        # local clean and filtering
        ents, ent_mapping = self._clean_and_deduplicate(ents)
        rels, rel_mapping = self._clean_and_deduplicate(rels)

        # LLM based entity and relation resolution
        # TODO: explore others methods for resolution (like lighter specialized models)
        # we want more shots for relations as they usually are more ambiguous
        new_ents, new_ent_mapping = self._iterative_resolution(
            ents, num_iters=5)
        new_rels, new_rel_mapping = self._iterative_resolution(
            rels, num_iters=5)

        consolidated_ent_mapping = {}
        for orig_id, cleaned_id in ent_mapping.items():
            final_id = new_ent_mapping[cleaned_id]
            consolidated_ent_mapping[orig_id] = final_id
        consolidated_rel_mapping = {}
        for orig_id, cleaned_id in rel_mapping.items():
            final_id = new_rel_mapping[cleaned_id]
            consolidated_rel_mapping[orig_id] = final_id

        ent_mapping = consolidated_ent_mapping
        rel_mapping = consolidated_rel_mapping
        ents = new_ents
        rels = new_rels

        triples = []
        for i in sorted(triples_by_ids.keys()):
            h, r, t = triples_by_ids[i]
            triple = (ents[ent_mapping[h]], rels[rel_mapping[r]],
                      ents[ent_mapping[t]])
            triples.append(triple)

        return triples


def _multistage_proc_helper(rank, in_chunks_per_proc, actions: List[Action],
                            remote_llm_caller: RemoteLLMCaller):
    per_chunk_results = []
    try:
        for chunk in in_chunks_per_proc[rank]:
            out = chunk
            for action in actions:
                try:
                    if isinstance(action, RemoteAction):
                        messages = action.prepare_prompt(chunk, out)
                        out = remote_llm_caller(messages)
                    out = action.parse(out)
                except Exception as e:
                    print(
                        f"[_multistage_proc_helper] Process {rank} failed on chunk processing: {type(e).__name__}: {str(e)}"
                    )
                    import traceback
                    print(
                        f"[_multistage_proc_helper] Process {rank} traceback: {traceback.format_exc()}"
                    )
                    out = []
                    break
            per_chunk_results += out
        torch.save(per_chunk_results, "/tmp/txt2kg_outs_for_proc_" + str(rank))
    except Exception as e:
        print(
            f"[_multistage_proc_helper] Process {rank} failed completely: {type(e).__name__}: {str(e)}"
        )
        import traceback
        print(
            f"[_multistage_proc_helper] Process {rank} complete failure traceback: {traceback.format_exc()}"
        )


def consume_actions(chunks: Tuple[str], actions: List[Action],
                    remote_llm_caller: RemoteLLMCaller) -> Any:
    result = []

    num_procs = min(len(chunks), _get_num_procs())
    meta_chunk_size = int(len(chunks) / num_procs)
    in_chunks_per_proc = {
        j:
        chunks[j * meta_chunk_size:min((j + 1) * meta_chunk_size, len(chunks))]
        for j in range(num_procs)
    }
    total_num_tries = 0
    for retry in range(5):
        try:
            for retry in range(200):
                try:
                    # debug
                    #for i in range(num_procs):
                    #    _multistage_proc_helper(i, in_chunks_per_proc, actions, remote_llm_caller)
                    mp.spawn(
                        _multistage_proc_helper,
                        args=(in_chunks_per_proc, actions, remote_llm_caller),
                        nprocs=num_procs, join=True)
                    break
                except Exception as e:
                    total_num_tries += 1
                    print(
                        f"[consume_actions] Process spawn failed on attempt {total_num_tries}: {type(e).__name__}: {str(e)}"
                    )
                    # For debugging, you might also want to see the full traceback:
                    import traceback
                    print(
                        f"[consume_actions] Full traceback: {traceback.format_exc()}"
                    )

            for rank in range(num_procs):
                result += torch.load(f"/tmp/txt2kg_outs_for_proc_{rank}")
                os.remove(f"/tmp/txt2kg_outs_for_proc_{rank}")
            break
        except Exception as e:
            total_num_tries += 1
            print(
                f"[consume_actions] Overall retry {retry+1}/5 failed: {type(e).__name__}: {str(e)}"
            )
            import traceback
            print(
                f"[consume_actions] Full traceback: {traceback.format_exc()}")
    print(f"[consume_actions] Total number of tries: {total_num_tries}")
    return result


def triples_to_nx(triples: List[Tuple[str, str, str]]) -> nx.Graph:
    graph = nx.Graph()
    for h, r, t in triples:
        graph.add_edge(h, t, relation=r)
    return graph


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

        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # (TODO) explore
        # self.embedding_model = SentenceTransformer(
        #     'Alibaba-NLP/gte-modernbert-base', device=device)

        self.remote_llm_caller = RemoteLLMCaller(NVIDIA_NIM_MODEL,
                                                 NVIDIA_API_KEY, ENDPOINT_URL)

        self.entity_extractor = LLMEntityExtractor()
        self.entity_augmenter = EntityAugmenter()
        self.triple_extractor = LLMTripleExtractor(
            system_prompt=TRIPLES_SYS_PROMPT_NEW)

        self.entity_resolver = LLMEntityResolver(self.remote_llm_caller)


    def save_kg(self, path: str) -> None:
        """Saves the relevant triples in the knowledge graph (KG) to a file.

        Args:
            path (str): The file path where the KG will be saved.

        Returns:
            None
        """
        tmp_path = path + ".tmp"
        torch.save(self.relevant_triples, tmp_path)
        os.rename(tmp_path, path)

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

    def reset(self):
        self.relevant_triples = {}

    def add_doc_2_KG(self, txt: str,
                         QA_pair: Optional[Tuple[str, str]] = None) -> None:
        """Add a document to the Knowledge Graph (KG).
        Args:
            txt (str): The text to extract triples from.
            QA_pair (Tuple[str, str]], optional):
                A QA pair to associate with the extracted triples.
                Useful for downstream evaluation.
        """
        # todo add qa pair handling later
        key = self.doc_id_counter
        self.doc_id_counter += 1

        # 1. Chunk wise extractions

        chunks = _chunk_text(txt, chunk_size=self.chunk_size)

        actions = [
            self.entity_extractor,
            self.entity_augmenter,
            self.triple_extractor,
        ]

        triples = consume_actions(chunks, actions, self.remote_llm_caller)

        # 2. Document wise linking

        triples = self.entity_resolver(triples)

        self.relevant_triples[key] = triples


    def add_doc_2_KG_old(
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

        # Handle empty text (context-less QA pairs)
        if txt == "":
            self.relevant_triples[key] = []
        else:
            # Chunk the text into smaller pieces for processing
            chunks = _chunk_text(txt, chunk_size=self.chunk_size)

            if self.local_LM:
                # For debugging purposes...
                # process chunks sequentially on the local LM
                self.relevant_triples[key] = _llm_then_python_parse(
                    chunks, _parse_n_check_triples,
                    self._chunk_to_triples_str_local)
            else:
                try:
                    num_procs = min(len(chunks), _get_num_procs())
                    self.relevant_triples[key] = _llm_call_and_consume(
                        chunks, SYSTEM_PROMPT, self.NVIDIA_API_KEY,
                        self.NIM_MODEL, self.ENDPOINT_URL, num_procs,
                        _parse_n_check_triples)
                except ImportError:
                    print(
                        "Failed to import `openai` package, please install it and rerun the script"
                    )
                    sys.exit(1)
        # Increment the doc_id_counter for the next document
        self.doc_id_counter += 1


# ##############################################################################
# ############################## Helper functions ##############################
# ##############################################################################


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


# ##############################################################################
# ############################## Legacy code ###################################
# ##############################################################################


def _llm_call_and_consume(chunks: Tuple[str], system_prompt: str,
                          NVIDIA_API_KEY: str, NIM_MODEL: str,
                          ENDPOINT_URL: str, num_procs: int,
                          post_process_fn: Callable[[str], List[Any]]) -> Any:
    result = []
    # Ensure NVIDIA_API_KEY is set before proceeding
    assert NVIDIA_API_KEY != '', \
        "Please init TXT2KG w/ NVIDIA_API_KEY or set local_lm=True"
    num_procs = min(len(chunks), _get_num_procs())
    meta_chunk_size = int(len(chunks) / num_procs)
    in_chunks_per_proc = {
        j:
        chunks[j * meta_chunk_size:min((j + 1) * meta_chunk_size, len(chunks))]
        for j in range(num_procs)
    }
    total_num_tries = 0
    for retry in range(5):
        try:
            for retry in range(200):
                try:
                    #_multiproc_helper(
                    #    0,
                    #    in_chunks_per_proc,
                    #    post_process_fn,
                    #    _chunk_to_result_cloud,
                    #    NVIDIA_API_KEY, NIM_MODEL,
                    #    ENDPOINT_URL, system_prompt)
                    mp.spawn(
                        _multiproc_helper,
                        args=(in_chunks_per_proc, post_process_fn,
                              _chunk_to_result_cloud, NVIDIA_API_KEY,
                              NIM_MODEL, ENDPOINT_URL, system_prompt),
                        nprocs=num_procs, join=True)
                    break
                except Exception as e:
                    total_num_tries += 1
                    print(
                        f"[_llm_call_and_consume] Process spawn failed on attempt {total_num_tries}: {type(e).__name__}: {str(e)}"
                    )
                    # For debugging, you might also want to see the full traceback:
                    import traceback
                    print(
                        f"[_llm_call_and_consume] Full traceback: {traceback.format_exc()}"
                    )

            for rank in range(num_procs):
                result += torch.load(f"/tmp/txt2kg_outs_for_proc_{rank}")
                os.remove(f"/tmp/txt2kg_outs_for_proc_{rank}")
            break
        except Exception as e:
            total_num_tries += 1
            print(
                f"[_llm_call_and_consume] Overall retry {retry+1}/5 failed: {type(e).__name__}: {str(e)}"
            )
            import traceback
            print(
                f"[_llm_call_and_consume] Full traceback: {traceback.format_exc()}"
            )
    print(f"[_llm_call_and_consume] Total number of tries: {total_num_tries}")
    return result


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
    if "llama-3.1-nemotron-ultra-253b-v1" in NIM_MODEL \
            or "kimi-k2-instruct" in NIM_MODEL:
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

def _chunk_to_result_cloud(txt: str, GLOBAL_NIM_KEY='',
                           NIM_MODEL="nvidia/llama-3.1-nemotron-70b-instruct",
                           ENDPOINT_URL="https://integrate.api.nvidia.com/v1",
                           system_prompt=SYSTEM_PROMPT) -> str:
    global CLIENT_INITD
    if not CLIENT_INITD:
        # We use NIMs since most PyG users may not be able to run a 70B+ model
        from openai import OpenAI
        global CLIENT
        CLIENT = OpenAI(base_url=ENDPOINT_URL, api_key=GLOBAL_NIM_KEY)
        CLIENT_INITD = True
    txt_input = txt
    completion = CLIENT.chat.completions.create(
        model=NIM_MODEL, messages=[{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": txt_input
        }], temperature=0, top_p=1, max_tokens=1024, stream=True)
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


def _multiproc_helper(rank, in_chunks_per_proc, py_fn, llm_fn, NIM_KEY,
                      NIM_MODEL, ENDPOINT_URL, system_prompt):
    out = _llm_then_python_parse(in_chunks_per_proc[rank], py_fn, llm_fn,
                                 GLOBAL_NIM_KEY=NIM_KEY, NIM_MODEL=NIM_MODEL,
                                 ENDPOINT_URL=ENDPOINT_URL,
                                 system_prompt=system_prompt)
    torch.save(out, "/tmp/txt2kg_outs_for_proc_" + str(rank))

