from torch_geometric.testing import onlyOnline, withPackage


@onlyOnline
@withPackage("pandas")
def test_ndc_classes_dataset(get_dataset) -> None:
    train_dataset = get_dataset(name="NDC-classes-25", split="train")
    val_dataset = get_dataset(name="NDC-classes-25", split="val")
    test_dataset = get_dataset(name="NDC-classes-25", split="test")

    assert str(train_dataset) == "CornellTemporalHyperGraphDatasets(34821)"
    assert train_dataset.num_features == 1

    assert str(val_dataset) == "CornellTemporalHyperGraphDatasets(7446)"
    assert val_dataset.num_features == 1

    assert str(test_dataset) == "CornellTemporalHyperGraphDatasets(7457)"
    assert test_dataset.num_features == 1

    # source: https://www.cs.cornell.edu/~arb/data/NDC-classes/
    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == 49724
