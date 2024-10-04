import re
from typing import List, Tuple

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


def find_weird_names(func_name_tokens, raw_dataset):
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
    return None


def make_df_from_raw_data(raw_dataset) -> DataFrame:
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
    """
    def __init__(
        self,
        root: str = "",
        split: str = "train",
        force_reload: bool = False,
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
        super().__init__(root, force_reload=force_reload)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def get_raw_python_from_df(self, func_name_tokens) -> Tuple[str, str]:
        # the ordering of code_search_net does not match ogbg-code2
        # have to search for matching python
        func_name = self.df.index
        basic_matches = (func_name == ''.join(func_name_tokens).lower()) | (
            func_name == '_'.join(func_name_tokens).lower()) | (
                func_name == "_" + "_".join(func_name_tokens).lower())

        basic_matches = basic_matches | (func_name
                                         == ''.join(func_name_tokens).lower())
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
            basic_matches = basic_matches | (func_name == '_'.join(
                func_name_tokens[:-1]).lower() + func_name_tokens[-1].lower())
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
            func_str = find_weird_names(func_name_tokens, self.combined_rawset)
            if func_str is None:
                # raw python data is missing from raw huggingface mirror
                # return empty strings
                return "", ""
        # save th func_signature and then the rest of function seperately
        return func_str[func_str.find("def"):(
            func_str.find(":"))], func_str[func_str.find('"""'):]

    def process(self) -> None:
        self.ogbg_dataset = PygGraphPropPredDataset(name="ogbg-code2")
        raw_datasets = datasets.load_dataset("claudios/code_search_net",
                                             "python")
        self.combined_rawset = datasets.concatenate_datasets([
            raw_datasets["train"], raw_datasets["validation"],
            raw_datasets["test"]
        ])
        self.df = make_df_from_raw_data(self.combined_rawset)
        for (split_name,
             idxs), path in zip(self.ogbg_dataset.get_idx_split().items(),
                                self.processed_paths):
            new_set = []
            for idx in tqdm(idxs, desc=split_name + "_preproc"):
                old_obj = self.ogbg_dataset[idx]
                new_obj = Data()
                # combine all node information into a single feature tensor, let the GNN+LLM figure it out
                new_obj.x = torch.cat(
                    (old_obj.x, old_obj.node_is_attributed,
                     old_obj.node_dfs_order, old_obj.node_depth),
                    dim=1).to(torch.float)
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
            self.save(new_set, path)
