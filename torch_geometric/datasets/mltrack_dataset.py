import os
import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from trackml.dataset import load_event
from torch_geometric.utils import process_event


class MLTrackDataset(InMemoryDataset):

    url = "https://www.kaggle.com/c/trackml-particle-identification/"

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(MLTrackDataset, self).__init__(root, transform, pre_transform,
                                             pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train_1.zip', 'train_2.zip', 'train_3.zip',
                'train_4.zip', 'train_5.zip']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} '
            'and move the zip files to {}'.
            format(self.raw_file_names, self.url, self.raw_dir))

    def process(self):
        for raw_path in self.raw_paths:
            # skip if already extracted, assuming the extraction was successful
            if not os.path.isdir(raw_path[:-4]):
                extract_zip(raw_path, self.raw_dir, log=False)

        processed_events = []
        for raw_path in self.raw_paths:
            raw_dir = raw_path[:-4]
            files = sorted(os.listdir(raw_dir))
            assert len(files) % 4 == 0, "some files might be missing"

            for i in range(0, len(files), 4):
                # removing suffix -cells.csv
                event_name = files[i][:-10]
                event_id = int(event_name[5:])
                load_path = os.path.join(self.processed_dir,
                                         'tmp_{}.pt'.format(event_name))
                try:
                    graphs = torch.load(load_path)
                    print('loaded', event_name)
                except FileNotFoundError:
                    print('processing', event_name)
                    try:
                        event_path = os.path.join(raw_dir, event_name)
                        hits, cells, particles, truth = load_event(event_path)
                        event = (event_id, hits, cells, particles, truth)
                        graphs = process_event(event)
                        torch.save(graphs, load_path)
                    except ValueError:
                        print('Couldn\'t parse {}'.format(event_name))

                processed_events.extend(graphs)
        torch.save(self.collate(processed_events), self.processed_paths[0])

        # deleting tmp files
        for file in os.listdir(self.processed_dir):
            if file.startswith('tmp'):
                os.remove(os.path.join(self.processed_dir, file))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


if __name__ == '__main__':
    dataset = MLTrackDataset('~/trackml_geometric_dataset')
    print(dataset)
