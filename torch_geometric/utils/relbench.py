from typing import Any

import numpy as np
import torch

from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index


def from_relbench(db: Any) -> HeteroData:
    r"""Converts a :class:`relbench.base.Database` object into a
    :class:`~torch_geometric.data.HeteroData` object.

    Each table in the database becomes a node type and each foreign key
    relationship becomes a bidirectional edge type.

    Numeric columns (excluding primary key, foreign key, and time columns)
    are concatenated into a node feature tensor :obj:`x`. If a table contains
    a time column, it is stored as a :obj:`time` attribute.

    Args:
        db (relbench.base.Database): A RelBench database instance containing
            a dictionary of tables linked by primary-foreign key
            relationships.

    Returns:
        HeteroData: A heterogeneous graph where each table maps to a node
        type and each foreign key relationship maps to a pair of directed
        edge types.

    Examples:
        >>> from relbench.base import Database, Table
        >>> import pandas as pd
        >>> users = Table(
        ...     df=pd.DataFrame({'id': [0, 1, 2], 'age': [25, 30, 35]}),
        ...     fkey_col_to_pkey_table={},
        ...     pkey_col='id',
        ... )
        >>> posts = Table(
        ...     df=pd.DataFrame({
        ...         'id': [0, 1, 2],
        ...         'user_id': [0, 1, 0],
        ...         'score': [10, 20, 30],
        ...     }),
        ...     fkey_col_to_pkey_table={'user_id': 'users'},
        ...     pkey_col='id',
        ... )
        >>> db = Database(table_dict={'users': users, 'posts': posts})
        >>> data = from_relbench(db)
        >>> data.node_types
        ['users', 'posts']
    """
    data = HeteroData()

    for table_name, table in db.table_dict.items():
        df = table.df

        # Determine columns to exclude from node features:
        exclude_cols = set()
        if table.pkey_col is not None:
            exclude_cols.add(table.pkey_col)
        if table.time_col is not None:
            exclude_cols.add(table.time_col)
        for fkey_col in table.fkey_col_to_pkey_table:
            exclude_cols.add(fkey_col)

        # Set number of nodes:
        data[table_name].num_nodes = len(df)

        # Convert numeric feature columns into a node feature tensor:
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype.kind in ('i', 'f')
        ]
        if feature_cols:
            x_np = df[feature_cols].to_numpy(
                dtype=np.float32,
                na_value=np.nan,
            )
            data[table_name].x = torch.from_numpy(x_np)

        # Store time column as Unix timestamp tensor:
        if table.time_col is not None:
            time_ser = df[table.time_col]
            if time_ser.dtype in [
                    np.dtype('datetime64[s]'),
                    np.dtype('datetime64[ns]'),
            ]:
                unix_time = time_ser.astype('int64').values
                if time_ser.dtype == np.dtype('datetime64[ns]'):
                    unix_time = unix_time // 10**9
                data[table_name].time = torch.from_numpy(unix_time)
            else:
                data[table_name].time = torch.from_numpy(
                    time_ser.values.astype(np.float64))

        # Create edges from foreign key relationships:
        for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_col]

            # Filter out dangling (NaN) foreign keys:
            mask = ~pkey_index.isna()
            fkey_idx = torch.arange(len(pkey_index))
            pkey_idx = torch.from_numpy(
                pkey_index[mask].to_numpy(dtype=np.int64))
            fkey_idx = fkey_idx[torch.from_numpy(mask.to_numpy(dtype=bool))]

            # Forward edge: fkey table -> pkey table
            edge_index = torch.stack([fkey_idx, pkey_idx], dim=0)
            edge_type = (table_name, f'f2p_{fkey_col}', pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # Reverse edge: pkey table -> fkey table
            edge_index = torch.stack([pkey_idx, fkey_idx], dim=0)
            edge_type = (pkey_table_name, f'rev_f2p_{fkey_col}', table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

    return data
