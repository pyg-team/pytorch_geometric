from types import SimpleNamespace
from typing import Any, Optional

import torch

from torch_geometric.testing import withPackage
from torch_geometric.utils import from_relbench


def _mock_table(
    df: Any,
    fkey_col_to_pkey_table: dict,
    pkey_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> SimpleNamespace:
    """Create a mock object that duck-types relbench.base.Table."""
    return SimpleNamespace(
        df=df,
        fkey_col_to_pkey_table=fkey_col_to_pkey_table,
        pkey_col=pkey_col,
        time_col=time_col,
    )


def _mock_database(table_dict: dict) -> SimpleNamespace:
    """Create a mock object that duck-types relbench.base.Database."""
    return SimpleNamespace(table_dict=table_dict)


@withPackage('pandas')
def test_from_relbench():
    import pandas as pd

    df_users = pd.DataFrame({
        'id': [0, 1, 2],
        'age': [25, 30, 35],
        'score': [1.0, 2.0, 3.0],
    })
    df_posts = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'user_id': [0, 1, 0, 2],
        'length': [100, 200, 150, 300],
    })

    users = _mock_table(
        df=df_users,
        fkey_col_to_pkey_table={},
        pkey_col='id',
    )
    posts = _mock_table(
        df=df_posts,
        fkey_col_to_pkey_table={'user_id': 'users'},
        pkey_col='id',
    )

    db = _mock_database(table_dict={'users': users, 'posts': posts})
    data = from_relbench(db)

    # Verify node types:
    assert 'users' in data.node_types
    assert 'posts' in data.node_types

    # Verify node counts:
    assert data['users'].num_nodes == 3
    assert data['posts'].num_nodes == 4

    # Verify numeric features were extracted:
    assert data['users'].x is not None
    assert data['users'].x.size() == (3, 2)  # age, score
    assert data['posts'].x is not None
    assert data['posts'].x.size() == (4, 1)  # length

    # Verify feature values:
    assert torch.allclose(
        data['users'].x,
        torch.tensor([[25, 1.0], [30, 2.0], [35, 3.0]]),
    )

    # Verify edge types (bidirectional fkey edges):
    edge_types = data.edge_types
    assert ('posts', 'f2p_user_id', 'users') in edge_types
    assert ('users', 'rev_f2p_user_id', 'posts') in edge_types

    # Verify edge index shapes (4 posts, each referencing a user):
    fwd = data['posts', 'f2p_user_id', 'users'].edge_index
    rev = data['users', 'rev_f2p_user_id', 'posts'].edge_index
    assert fwd.size() == (2, 4)
    assert rev.size() == (2, 4)


@withPackage('pandas')
def test_from_relbench_dangling_fkeys():
    """Test that dangling (NaN) foreign keys are filtered out."""
    import pandas as pd

    df_users = pd.DataFrame({'id': [0, 1]})
    df_posts = pd.DataFrame({
        'id': [0, 1, 2],
        'user_id':
        pd.array([0, None, 1], dtype=pd.Int64Dtype()),
    })

    users = _mock_table(
        df=df_users,
        fkey_col_to_pkey_table={},
        pkey_col='id',
    )
    posts = _mock_table(
        df=df_posts,
        fkey_col_to_pkey_table={'user_id': 'users'},
        pkey_col='id',
    )

    db = _mock_database(table_dict={'users': users, 'posts': posts})
    data = from_relbench(db)

    # Only 2 out of 3 posts have valid foreign keys:
    fwd = data['posts', 'f2p_user_id', 'users'].edge_index
    assert fwd.size() == (2, 2)


@withPackage('pandas')
def test_from_relbench_time_column():
    """Test that time columns are correctly converted."""
    import pandas as pd

    df = pd.DataFrame({
        'id': [0, 1, 2],
        'ts':
        pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'val': [10, 20, 30],
    })

    events = _mock_table(
        df=df,
        fkey_col_to_pkey_table={},
        pkey_col='id',
        time_col='ts',
    )

    db = _mock_database(table_dict={'events': events})
    data = from_relbench(db)

    assert data['events'].num_nodes == 3
    assert data['events'].time is not None
    assert data['events'].time.size() == (3, )
    # Time column should not appear in features:
    assert data['events'].x.size() == (3, 1)  # only 'val'


@withPackage('pandas')
def test_from_relbench_no_features():
    """Test tables with only pkey/fkey columns and no numeric features."""
    import pandas as pd

    df = pd.DataFrame({
        'id': [0, 1, 2],
        'name': ['a', 'b', 'c'],  # Non-numeric, should be excluded
    })

    items = _mock_table(
        df=df,
        fkey_col_to_pkey_table={},
        pkey_col='id',
    )

    db = _mock_database(table_dict={'items': items})
    data = from_relbench(db)

    assert data['items'].num_nodes == 3
    # No numeric feature columns (name is string, id is pkey):
    assert not hasattr(data['items'], 'x') or data['items'].x is None


@withPackage('relbench')
def test_from_relbench_with_relbench():
    """Integration test using actual relbench objects."""
    import pandas as pd
    from relbench.base import Database, Table

    df_users = pd.DataFrame({
        'id': [0, 1, 2],
        'age': [25, 30, 35],
    })
    df_posts = pd.DataFrame({
        'id': [0, 1, 2],
        'user_id': [0, 1, 0],
        'score': [10, 20, 30],
    })

    users = Table(
        df=df_users,
        fkey_col_to_pkey_table={},
        pkey_col='id',
    )
    posts = Table(
        df=df_posts,
        fkey_col_to_pkey_table={'user_id': 'users'},
        pkey_col='id',
    )

    db = Database(table_dict={'users': users, 'posts': posts})
    data = from_relbench(db)

    assert 'users' in data.node_types
    assert 'posts' in data.node_types
    assert data['users'].num_nodes == 3
    assert data['posts'].num_nodes == 3
