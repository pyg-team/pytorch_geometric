# torch_geometric/datasets/echo.py

from __future__ import annotations

import os
from typing import Callable, Dict, List, Literal, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


Task = Literal["sssp", "diam", "ecc", "energy", "charge"]
Split = Literal["train", "val", "test"]


class ECHOBenchmark(InMemoryDataset):
    r"""The **ECHOBenchmark** dataset introduced in
    `"Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation"
    <https://openreview.net/forum?id=DgkWFPZMPp>`_.

    ECHO is a benchmark for evaluating long-range graph propagation
    capabilities of graph neural networks. It contains five graph
    tasks:

    - ``sssp`` (Single-Source Shortest Path)
    - ``diam`` (Graph Diameter)
    - ``ecc`` (Node Eccentricity)
    - ``energy`` (Molecular Energy)
    - ``charge`` (Molecular Partial Charge)

    Each task provides three predefined splits (``train``, ``val``,
    ``test``), where every split consists of a list of homogeneous
    :class:`torch_geometric.data.Data` objects.

    The dataset is hosted on HuggingFace at:
    https://huggingface.co/datasets/gmander44/echo

    Args:
        root (str): Root directory where the dataset should be saved.
        task (str): One of
            ``{'sssp', 'diam', 'ecc', 'energy', 'charge'}``.
        split (str): One of ``{'train', 'val', 'test'}``.
        transform (callable, optional): A function/transform that takes in a
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    Example:
        >>> from torch_geometric.datasets import ECHOBenchmark
        >>> train = ECHOBenchmark(root='data/ECHOBenchmark', task='sssp', split='train')
        >>> len(train)
        >>> data = train[0]

    **Citation:**

    .. code-block:: bibtex

        @inproceedings{echobenchmark,
        title={Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation},
        author={Miglior, Luca and Tolloso, Matteo and Gravina, Alessio and Bacciu, Davide},
        booktitle={The Fourteenth International Conference on Learning Representations},
        year={2026},
        url={https://openreview.net/forum?id=DgkWFPZMPp}
        }
    """


    URLS: Dict[str, Dict[str, str]] = {
        "sssp": {
            "train": "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/sssp/train_data.pt",
            "val":   "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/sssp/val_data.pt",
            "test":  "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/sssp/test_data.pt",
        },
        "diam": {
            "train": "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/diam/train_data.pt",
            "val":   "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/diam/val_data.pt",
            "test":  "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/diam/test_data.pt",
        },
        "ecc": {
            "train": "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/ecc/train_data.pt",
            "val":   "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/ecc/val_data.pt",
            "test":  "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/ecc/test_data.pt",
        },
        "energy": {
            "train": "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/energy/train_data.pt",
            "val":   "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/energy/val_data.pt",
            "test":  "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-synth/energy/test_data.pt",
        },
        "charge": {
            "train": "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-charge/train_data.pt",
            "val":   "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-charge/val_data.pt",
            "test":  "https://huggingface.co/datasets/gmander44/echo/resolve/main/echo-charge/test_data.pt",
        },
    }

    tasks = sorted(URLS.keys())
    splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str,
        task: Task,
        split: Split,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.task = str(task)
        self.split = str(split)

        if self.task not in self.tasks:
            raise ValueError(
                f"Invalid task '{self.task}'. Expected one of {self.tasks}."
            )
        if self.split not in self.splits:
            raise ValueError(
                f"Invalid split '{self.split}'. Expected one of {self.splits}."
            )

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        path = self.processed_paths[self._processed_index()]
        self.load(path)

    @property
    def raw_dir(self) -> str:
        # Put ECHOBenchmark raw files under a subfolder to keep root tidy.
        return os.path.join(self.root, "raw", "echobenchmark", self.task)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed", "echobenchmark", self.task)

    @property
    def processed_file_names(self) -> List[str]:
        # 3 processed artifacts per task
        return [f"{self.task}_{s}.pt" for s in self.splits]

    def _processed_index(self) -> int:
        split_to_idx = {"train": 0, "val": 1, "test": 2}
        return split_to_idx[self.split]

    @property
    def raw_file_names(self) -> List[str]:
        # Standard split filenames inside raw_dir:
        return [f"{s}.pt" for s in self.splits]

    def download(self) -> None:
        urls_for_task = self.URLS.get(self.task)
        if urls_for_task is None:
            raise RuntimeError(
                f"Missing URLs for task '{self.task}' in ECHOBenchmark.URLS"
            )

        os.makedirs(self.raw_dir, exist_ok=True)

        for s in self.splits:
            url = urls_for_task.get(s)
            if url is None:
                raise RuntimeError(
                    f"Missing URL for task='{self.task}', split='{s}' in ECHOBenchmark.URLS"
                )

            # Download into raw_dir (keeps remote filename, e.g. train_data.pt):
            download_url(url, self.raw_dir)

            # Compute expected downloaded filename from the URL and rename it:
            basename = url.split("/")[-1].split("?")[0]  # train_data.pt
            src = os.path.join(self.raw_dir, basename)
            dst = os.path.join(self.raw_dir, f"{s}.pt")  # train.pt

            if not os.path.exists(src):
                # As a fallback, if HF/redirect changed filename, try to find a single *.pt:
                candidates = [p for p in os.listdir(self.raw_dir) if p.endswith(".pt")]
                raise FileNotFoundError(
                    f"Expected downloaded file '{src}' not found. Found: {candidates}"
                )

            if src != dst:
                os.replace(src, dst)

    def process(self) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)

        for s in self.splits:
            raw_path = os.path.join(self.raw_dir, f"{s}.pt")
            try:
                data_list = torch.load(raw_path, weights_only=False)
            except TypeError:
                data_list = torch.load(raw_path)

            if not isinstance(data_list, list):
                raise TypeError(
                    f"ECHOBenchmark expected a list of Data in '{raw_path}', got {type(data_list)}"
                )

            for i, item in enumerate(data_list[:5]):
                if not isinstance(item, Data):
                    raise TypeError(
                        f"ECHOBenchmark expected list[Data] in '{raw_path}', but element {i} is {type(item)}"
                    )

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            out_path = os.path.join(self.processed_dir, f"{self.task}_{s}.pt")
            self.save(data_list, out_path)