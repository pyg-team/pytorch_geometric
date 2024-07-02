from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_web_qsp_dataset(get_dataset):
	dataset = get_dataset(name='WebQSPDataset')
	assert len(dataset) == 4700
	assert str(dataset) == "WebQSPDataset(4700)"
	