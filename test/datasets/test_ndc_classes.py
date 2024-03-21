from torch_geometric.testing import onlyOnline


@onlyOnline
def test_ndc_classes_dataset(get_dataset) -> None:
    train_dataset = get_dataset(name="NDCClasses25", split="train")
    val_dataset = get_dataset(name="NDCClasses25", split="val")
    test_dataset = get_dataset(name="NDCClasses25", split="test")

    assert str(train_dataset) == "NDCClasses25(34821)"
    assert train_dataset.num_features == 1

    assert str(val_dataset) == "NDCClasses25(7446)"
    assert val_dataset.num_features == 1

    assert str(test_dataset) == "NDCClasses25(7457)"
    assert test_dataset.num_features == 1

    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == 49724
