import torch
from torch_geometric.nn import knn_interpolate


def test_knn_interpolate():
    x_pos = torch.tensor([[-3, 0], [0, 0], [3, 0],
                          [-2, 0], [0, 0], [2, 0]], dtype=torch.float)
    x_x = torch.tensor([[1], [10], [100],
                        [-1], [-10], [-100]], dtype=torch.float)
    batch_x = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    y_pos = torch.tensor([[-2, -1], [2, 1],
                          [-3, -1], [3, 1]], dtype=torch.float)
    y_x = torch.tensor([[1], [2],
                        [3], [4]], dtype=torch.float)
    batch_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    y_x_new_2_true = torch.tensor([[25. / 7.], [520. / 7.],
                                   [-2.5], [-85]], dtype=torch.float)
    y_x_new_2_cat_true = torch.tensor([[1, 25. / 7.], [2, 520. / 7.],
                                       [3, -2.5], [4, -85]], dtype=torch.float)

    y_x_new_2 = knn_interpolate(x_pos, y_pos, x_x, y_x, batch_x, batch_y,
                                k=2, cat=False)
    y_x_new_2_cat = knn_interpolate(x_pos, y_pos, x_x, y_x, batch_x, batch_y,
                                    k=2, cat=True)

    y_x_new_3_true = torch.tensor([[825. / 96.], [6765. / 96.],
                                   [-695. / 83.], [-6635. / 83.]],
                                  dtype=torch.float)
    y_x_new_3_cat_true = torch.tensor([[1, 825. / 96.], [2, 6765. / 96.],
                                       [3, -695. / 83.], [4, -6635. / 83.]],
                                      dtype=torch.float)

    y_x_new_3 = knn_interpolate(x_pos, y_pos, x_x, y_x, batch_x, batch_y,
                                k=3, cat=False)
    y_x_new_3_cat = knn_interpolate(x_pos, y_pos, x_x, y_x, batch_x, batch_y,
                                    k=3, cat=True)

    assert torch.allclose(y_x_new_2, y_x_new_2_true, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y_x_new_2_cat, y_x_new_2_cat_true, rtol=1e-3,
                          atol=1e-3)
    assert torch.allclose(y_x_new_3, y_x_new_3_true, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y_x_new_3_cat, y_x_new_3_cat_true, rtol=1e-3,
                          atol=1e-3)
