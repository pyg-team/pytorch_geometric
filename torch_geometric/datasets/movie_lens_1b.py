import os
import os.path as osp
import re
from typing import Callable, List, Optional, Tuple

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_tar
)
from torch_geometric.io import fs


class MovieLens1B(InMemoryDataset):
    r"""The MovieLens 1B heterogeneous rating dataset, assembled by GroupLens
    Research from the `MovieLens web site <https://movielens.org>`__,
    consisting of movies (3,883 nodes) and users (6,040 nodes) with
    approximately 1 billion ratings between them.
    User ratings for movies are available as ground truth labels.
    Features of users and movies are encoded according to the `"Inductive
    Matrix Completion Based on Graph Neural Networks"
    <https://arxiv.org/abs/1904.12058>`__ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        add_reverse_edges (bool, optional): If set to :obj:`True`, also stores
            reverse training edges in :obj:`('movie', 'rated_by', 'user')`.
            (default: :obj:`False`)
        num_shards (int, optional): Number of NPZ shard files to load per
            split (train/test). Each shard contains ~1/16th of the data.
            Use a small value (e.g. 1 or 2) to reduce memory usage for
            quick experiments. If :obj:`None`, all 16 shards are loaded.
            (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10
        :header-rows: 1

        * - Node/Edge Type
          - #nodes/#edges
          - #features
          - #tasks
        * - Movie
          - 855,776
          - 
          -
        * - User
          - 2,210,078
          - 
          -
        * - User-Movie
          - 1,223,962,043
          - 
          - 
    """
    url = 'https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        add_reverse_edges: bool = False,
        num_shards: Optional[int] = None,
        force_reload: bool = False,
    ) -> None:
        self.add_reverse_edges = add_reverse_edges
        self.num_shards = num_shards
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            *(f'trainx16x32_{i}.npz' for i in range(16)),
            *(f'testx16x32_{i}.npz' for i in range(16)),
        ]

    @property
    def processed_file_names(self) -> str:
        if self.num_shards is not None:
            return f'data_shards{self.num_shards}.pt'
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_tar(path, self.root, mode='r')
        os.remove(path)
        folder = osp.join(self.root, 'ml-20mx16x32')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self) -> None:
        import numpy as np

        data = HeteroData()

        def shard_index(path: str) -> int:
            match = re.search(r'_(\d+)\.npz$', path)
            return int(match.group(1)) if match is not None else -1

        def get_shard_files(split: str, limit: Optional[int]) -> List[str]:
            if limit is not None and limit <= 0:
                raise ValueError(f'Expected positive shard limit, got {limit}')

            files = sorted(
                [
                    osp.join(self.raw_dir, name)
                    for name in os.listdir(self.raw_dir)
                    if name.startswith(f'{split}x16x32_') and name.endswith('.npz')
                ],
                key=shard_index,
            )
            if not files:
                raise FileNotFoundError(
                    f'No NPZ shard files found for split={split!r} in '
                    f'{self.raw_dir}'
                )
            return files if limit is None else files[:limit]

        def summarize_shards(paths: List[str]) -> Tuple[int, int, int, int]:
            total_rows = 0
            max_user_id = -1
            max_movie_id = -1
            num_cols = -1
            for path in paths:
                with np.load(path, allow_pickle=False) as z:
                    array = z['arr_0']
                if array.ndim != 2 or array.shape[1] < 2:
                    raise ValueError(
                        f'Expected shape [num_rows, >=2] in {path}, '
                        f'got {tuple(array.shape)}'
                    )
                total_rows += int(array.shape[0])
                max_user_id = max(max_user_id, int(array[:, 0].max()))
                max_movie_id = max(max_movie_id, int(array[:, 1].max()))
                if num_cols == -1:
                    num_cols = int(array.shape[1])
                else:
                    num_cols = min(num_cols, int(array.shape[1]))
            return total_rows, max_user_id, max_movie_id, num_cols

        def build_split_tensors(
            paths: List[str],
            total_rows: int,
            has_rating: bool,
            has_time: bool,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            edge_index = torch.empty((2, total_rows), dtype=torch.long)
            rating = torch.empty(total_rows, dtype=torch.float) if has_rating else None
            time = torch.empty(total_rows, dtype=torch.long) if has_time else None

            offset = 0
            for path in paths:
                with np.load(path, allow_pickle=False) as z:
                    array = z['arr_0']

                num_rows = int(array.shape[0])
                next_offset = offset + num_rows

                edge_index[:, offset:next_offset] = torch.from_numpy(
                    array[:, :2].T
                ).to(torch.long)

                if rating is not None:
                    rating[offset:next_offset] = torch.from_numpy(array[:, 2]).to(
                        torch.float
                    )
                if time is not None:
                    time[offset:next_offset] = torch.from_numpy(array[:, 3]).to(
                        torch.long
                    )

                offset = next_offset

            return edge_index, rating, time

        train_files = get_shard_files('train', self.num_shards)
        test_files = get_shard_files('test', self.num_shards)

        train_rows, train_max_user_id, train_max_movie_id, train_cols = summarize_shards(
            train_files
        )
        test_rows, test_max_user_id, test_max_movie_id, test_cols = summarize_shards(
            test_files
        )

        max_user_id = max(train_max_user_id, test_max_user_id)
        max_movie_id = max(train_max_movie_id, test_max_movie_id)

        data['user'].num_nodes = max_user_id + 1
        data['movie'].num_nodes = max_movie_id + 1

        train_edge_index, train_rating, train_time = build_split_tensors(
            train_files,
            train_rows,
            has_rating=(train_cols >= 3),
            has_time=(train_cols >= 4),
        )
        data['user', 'rates', 'movie'].edge_index = train_edge_index

        if train_rating is None:
            train_rating = torch.ones(train_rows, dtype=torch.float)
        data['user', 'rates', 'movie'].rating = train_rating

        if train_time is not None:
            data['user', 'rates', 'movie'].time = train_time

        data['movie', 'rated_by', 'user'].edge_index = train_edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = train_rating

        if train_time is not None:
            data['movie', 'rated_by', 'user'].time = train_time

        test_edge_label_index, test_edge_label, _ = build_split_tensors(
            test_files,
            test_rows,
            has_rating=(test_cols >= 3),
            has_time=False,
        )

        data['user', 'rates', 'movie'].edge_label_index = test_edge_label_index

        if test_edge_label is None:
            test_edge_label = torch.ones(test_rows, dtype=torch.float)
        data['user', 'rates', 'movie'].edge_label = test_edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
