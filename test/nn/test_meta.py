import unittest.mock as mock
from torch_geometric.nn.meta import MetaLayer


def test_MetaLayer_mock():
    edge_model = mock.MagicMock()
    node_model = mock.MagicMock()
    global_model = mock.MagicMock()

    for em in (edge_model, None):
        for nm in (node_model, None):
            for gm in (global_model, None):
                lay = MetaLayer(edge_model=em, node_model=nm, global_model=gm)
                out = lay(
                    x=mock.MagicMock(),
                    edge_index=("row", "col"),
                    edge_attr="edge_attr",
                    u="u",
                    batch="batch"
                )
                assert isinstance(out, tuple)
                assert len(out) == 3
                if em is not None:
                    em.assert_called_once()
                if nm is not None:
                    nm.assert_called_once()
                if gm is not None:
                    gm.assert_called_once()

                edge_model.reset_mock()
                node_model.reset_mock()
                global_model.reset_mock()
