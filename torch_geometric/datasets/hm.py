from typing import Callable, List, Optional

import torch

from torch_geometric.data import HeteroData, InMemoryDataset


class HM(InMemoryDataset):
    r"""The heterogeneous H&M dataset from the `Kaggle H&M Personalized Fashion
    Recommendations
    <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations>`_
    challenge.
    The task is to develop product recommendations based on data from previous
    transactions, as well as from customer and product meta data.

    Args:
        root (str): Root directory where the dataset should be saved.
        use_all_tables_as_node_types (bool, optional): If set to :obj:`True`,
            will use the transaction table as a distinct node type.
            (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = ('https://www.kaggle.com/competitions/'
           'h-and-m-personalized-fashion-recommendations/data')

    def __init__(
        self,
        root: str,
        use_all_tables_as_node_types: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.use_all_tables_as_node_types = use_all_tables_as_node_types
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'customers.csv.zip', 'articles.csv.zip',
            'transactions_train.csv.zip'
        ]

    @property
    def processed_file_names(self) -> str:
        if self.use_all_tables_as_node_types:
            return 'data.pt'
        else:
            return 'data_merged.pt'

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download {self.raw_file_names} from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Process customer data ###############################################
        df = pd.read_csv(self.raw_paths[0], index_col='customer_id')
        customer_map = {idx: i for i, idx in enumerate(df.index)}

        xs = []
        for name in [
                'Active', 'FN', 'club_member_status', 'fashion_news_frequency'
        ]:
            x = pd.get_dummies(df[name]).values
            xs.append(torch.from_numpy(x).to(torch.float))

        x = torch.from_numpy(df['age'].values).to(torch.float).view(-1, 1)
        x = x.nan_to_num(nan=x.nanmean())  # type: ignore
        xs.append(x / x.max())

        data['customer'].x = torch.cat(xs, dim=-1)

        # Process article data ################################################
        df = pd.read_csv(self.raw_paths[1], index_col='article_id')
        article_map = {idx: i for i, idx in enumerate(df.index)}

        xs = []
        for name in [  # We drop a few columns here that are high cardinality.
                # 'product_code',  # Drop.
                # 'prod_name',  # Drop.
                'product_type_no',
                'product_type_name',
                'product_group_name',
                'graphical_appearance_no',
                'graphical_appearance_name',
                'colour_group_code',
                'colour_group_name',
                'perceived_colour_value_id',
                'perceived_colour_value_name',
                'perceived_colour_master_id',
                'perceived_colour_master_name',
                # 'department_no',  # Drop.
                # 'department_name',  # Drop.
                'index_code',
                'index_name',
                'index_group_no',
                'index_group_name',
                'section_no',
                'section_name',
                'garment_group_no',
                'garment_group_name',
                # 'detail_desc',  # Drop.
        ]:
            x = pd.get_dummies(df[name]).values
            xs.append(torch.from_numpy(x).to(torch.float))

        data['article'].x = torch.cat(xs, dim=-1)

        # Process transaction data ############################################
        df = pd.read_csv(self.raw_paths[2], parse_dates=['t_dat'])

        x1 = pd.get_dummies(df['sales_channel_id']).values
        x1 = torch.from_numpy(x1).to(torch.float)
        x2 = torch.from_numpy(df['price'].values).to(torch.float).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        time = torch.from_numpy(df['t_dat'].values.astype(int))
        time = time // (60 * 60 * 24 * 10**9)  # Convert nanoseconds to days.

        src = torch.tensor([customer_map[idx] for idx in df['customer_id']])
        dst = torch.tensor([article_map[idx] for idx in df['article_id']])

        if self.use_all_tables_as_node_types:
            data['transaction'].x = x
            data['transaction'].time = time

            edge_index = torch.stack([src, torch.arange(len(df))], dim=0)
            data['customer', 'to', 'transaction'].edge_index = edge_index
            edge_index = edge_index.flip([0])
            data['transaction', 'rev_to', 'customer'].edge_index = edge_index

            edge_index = torch.stack([dst, torch.arange(len(df))], dim=0)
            data['article', 'to', 'transaction'].edge_index = edge_index
            edge_index = edge_index.flip([0])
            data['transaction', 'rev_to', 'article'].edge_index = edge_index
        else:
            edge_index = torch.stack([src, dst], dim=0)
            data['customer', 'to', 'article'].edge_index = edge_index
            data['customer', 'to', 'article'].time = time
            data['customer', 'to', 'article'].edge_attr = x

            edge_index = edge_index.flip([0])
            data['article', 'rev_to', 'customer'].edge_index = edge_index
            data['article', 'rev_to', 'customer'].time = time
            data['article', 'rev_to', 'customer'].edge_attr = x

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
