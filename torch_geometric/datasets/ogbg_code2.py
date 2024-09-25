import re
from typing import List

import torch
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset

try:
    from ogb.graphproppred import PygGraphPropPredDataset
    WITH_OGB = True
except ImportError:
    WITH_OGB = False

try:
    import datasets
    WITH_DATASETS = True
except ImportError:
    WITH_DATASETS = False

try:
    import cudf.pandas
    cudf.pandas.install()
    WITH_CUDF = True
except Exception:
    WITH_CUDF = False

try:
    # This needs to be done after cudf.pandas.install to use CUDF
    from pandas import DataFrame
    WITH_PANDAS = True
except ImportError:
    DataFrame = None
    WITH_PANDAS = False


def find_wierd_names(func_name_tokens, raw_dataset):
    for i, func_name in enumerate(raw_dataset["func_name"]):
        # helper code to find wierd matches
        # since its non-trivial to apply such complex search to pandas
        func_name = func_name.split('.')[-1].lower()
        escaped_tokens = [re.escape(item).lower() for item in func_name_tokens]
        # REGEX by ChatGPT
        pattern = r'(^|_)?(' + '|'.join(escaped_tokens) + r')(_|$)?'
        final_pattern = r'^(?:' + pattern + r')+$'
        regex = re.compile(final_pattern)
        if regex.match(func_name):
            return raw_dataset["whole_func_string"][i]
    raise ValueError("nothing found for func_name_tokens =", func_name_tokens)


def make_df_from_raw_data(raw_dataset):
    # Create a Data Frame with
    # column 1: "func_name"
    # column 2: "whole_func_string"
    filtered_func_names = []
    for i in raw_dataset["func_name"]:
        split_name = i.split('.')
        if len(split_name) > 1:
            filtered_func_names.append(split_name[-1].lower())
        else:
            filtered_func_names.append(i.lower())
    df = DataFrame({
        'func_name': filtered_func_names,
        "whole_func_string": raw_dataset["whole_func_string"]
    })
    df.set_index('func_name', inplace=True)
    return df


class OGBG_Code2(InMemoryDataset):
    r"""The ogbg-code2 dataset focuses on predicting the name of a python
    function, given an Abstract Syntax Tree (AST) of the function.
    See https://ogb.stanford.edu/docs/graphprop/#ogbg-code2 for details.
    This PyG mirror builds on the original dataset by adding the raw python
    tokens to the dataset so that users can train GNN+LLM systems to beat the
    current GNN only SOTA results in the leaderboard.

    Args:
        root (str): Root directory where the dataset should be saved.
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        include_raw_python (bool, optional): Whether to include raw python
            text alongside the graph data.
            (default: :obj:`True`)
    """
    def __init__(
        self,
        root: str = "",
        force_reload: bool = False,
        include_raw_python: bool = True,
    ) -> None:
        missing_str_list = []
        if not WITH_OGB:
            missing_str_list.append('ogb')
        if not WITH_DATASETS:
            missing_str_list.append('datasets')
        if not WITH_PANDAS:
            missing_str_list.append('pandas')
        if len(missing_str_list) > 0:
            missing_str = ' '.join(missing_str_list)
            error_out = f"`pip install {missing_str}` to use this dataset."
            raise ImportError(error_out)
        if not WITH_CUDF:
            print("Note: OGBG_Code2 PyG dataset uses pandas.")
            print("Install NVIDIA CUDF for massive speedups in preproc.")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(root, None, None, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["list_of_graphs.pt", "pre_filter.pt", "pre_transform.pt"]

    def download(self) -> None:
        self.ogbg_dataset = PygGraphPropPredDataset(name="ogbg-code2")
        dataset = datasets.load_dataset("claudios/code_search_net", "python")
        self.raw_dataset = datasets.concatenate_datasets(
            [dataset["train"], dataset["validation"], dataset["test"]])
        self.split_idxs = {
            "train":
            torch.arange(len(dataset["train"])),
            "val":
            torch.arange(len(dataset["validation"])) + len(dataset["train"]),
            "test":
            torch.arange(len(dataset["test"])) + len(dataset["train"]) +
            len(dataset["validation"])
        }
        self.df = make_df_from_raw_data(self.raw_dataset)

    def get_raw_python_from_df(self, func_name_tokens):
        # the ordering of code_search_net does not match ogbg-code2
        # have to search for matching python
        func_name = self.df.index
        basic_matches = (func_name == ''.join(func_name_tokens).lower()) | (
            func_name == '_'.join(func_name_tokens).lower()) | (
                func_name == "_" + "_".join(func_name_tokens).lower())
        basic_matches = basic_matches | (func_name == "_" +
                                         ''.join(func_name_tokens).lower())
        basic_matches = basic_matches | (
            func_name == "_" + ''.join(func_name_tokens).lower() + "_")
        basic_matches = basic_matches | (
            func_name == "_" + '_'.join(func_name_tokens).lower() + "_")
        basic_matches = basic_matches | (
            func_name == "_" + '_'.join(func_name_tokens).lower() + "_")
        if len(func_name_tokens) > 1:
            basic_matches = basic_matches | (
                func_name == "_" + func_name_tokens[0].lower() + "_" +
                ''.join(func_name_tokens[1:]).lower())
            basic_matches = basic_matches | (
                func_name == ''.join(func_name_tokens[:-1]).lower() + "_" +
                func_name_tokens[-1].lower())
            # basic_matches = basic_matches | (
            #     func_name == '_'.join(func_name_tokens[:-1]).lower() +
            #     func_name_tokens[-1].lower())
        matches = basic_matches
        result = self.df[matches]
        if len(result) > 0:
            """
            Randomly select one of the matches. We randomly select since we
                have no idea how to break these ties. If this proves to be a blocker for SOTA results,
                will come back to figure out how to do this best."""
            selected_result = result.sample()
            func_str = str(selected_result.iloc[0]["whole_func_string"])
        else:
            func_str = find_wierd_names(func_name_tokens, self.raw_dataset)
        return func_str[func_str.find("def"):(
            func_str.find(":"))], func_str[func_str.find('"""'):]

    def process(self) -> None:
        new_set = []
        len_set = len(self.ogbg_dataset)
        #print("num_data_pts =", len_set)
        for i in tqdm(range(len_set)):
            old_obj = self.ogbg_dataset[i]
            new_obj = Data()
            # combine all node information into a single feature tensor, let the GNN+LLM figure it out
            new_obj.x = torch.cat((old_obj.x, old_obj.node_is_attributed,
                                   old_obj.node_dfs_order, old_obj.node_depth),
                                  dim=1)
            # extract raw python function for use by LLM
            func_name_tokens = old_obj.y
            new_obj.func_signature, new_obj.desc = self.get_raw_python_from_df(
                func_name_tokens)
            # extract other data needed for GNN+LLM
            new_obj.y = old_obj.y
            new_obj.edge_index = old_obj.edge_index
            new_obj.num_nodes = old_obj.num_nodes
            new_set.append(new_obj)
            del old_obj
        self.save(new_set, self.processed_paths[0])
