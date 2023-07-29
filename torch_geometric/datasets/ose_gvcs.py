from typing import Callable, List, Optional

import torch
import os
from torch_geometric.data import InMemoryDataset, download_url, extract_tar, HeteroData

class OSE_GVCS(InMemoryDataset):
    r"""`Product ecology <https://wiki.opensourceecology.org/wiki/Product_Ecologies>`_ for Open Source Ecology's iconoclastic `Global Village Construction Set <https://wiki.opensourceecology.org/wiki/Global_Village_Construction_Set>`_.
    A modular, DIY, low-cost, high-performance platform that allows for the easy fabrication of the 50 different industrial machines that it takes to build a small, sustainable civilization with modern comforts.
    The 50 original type :obj:`"machine"` nodes, composing the GVCS.
    290 directed unweighted edges, each representing a relationship type between pairs of machines.
    Processing a minimalist heterogenous graph from JSON.
	"""

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    url = 'https://github.com/Wesxdz/ose_gvcs/raw/master/ose_gvcs.tar.gz'

    @property
    def raw_file_names(self):
        return [f"{machine.lower().replace(' ', '_')}.json" for machine in self.machines]

    @property
    def processed_file_names(self):
        return ['gvcs.pt']

    def download(self):
        path = download_url(self.url, self.root)
        extract_tar(path, f'{self.root}/raw')
        os.unlink(path)

    @property
    def machines(self) -> List[str]:
        return ['3D Printer', '3D Scanner', 'Aluminum Extractor', 'Backhoe', 'Bakery Oven', 
                'Baler', 'Bioplastic Extruder', 'Bulldozer', 'Car', 'CEB Press', 
                'Cement Mixer', 'Chipper Hammermill', 'CNC Circuit Mill', 'CNC Torch Table', 'Dairy Milker', 
                'Drill Press', 'Electric Motor Generator', 'Gasifier Burner', 'Hay Cutter', 'Hay Rake', 
                'Hydraulic Motor', 'Induction Furnace', 'Industrial Robot', 'Ironworker', 'Laser Cutter', 
                'Metal Roller', 'Microcombine', 'Microtractor', 'Multimachine', 'Nickel-Iron Battery', 
                'Pelletizer', 'Plasma Cutter', 'Power Cube', 'Press Forge', 'Rod and Wire Mill', 
                'Rototiller', 'Sawmill', 'Seeder', 'Solar Concentrator', 'Spader', 
                'Steam Engine', 'Steam Generator', 'Tractor', 'Trencher', 'Truck', 
                'Universal Power Supply', 'Universal Rotor', 'Welder', 'Well-Drilling Rig', 'Wind Turbine']
    
    # node feature 0
    categories = ['habitat', 'agriculture', 'industry', 'energy', 'materials', 'transportation']
    
    # edge relationship types
    relationships = ['from', 'uses', 'enables']

    def process(self):
        import json
        import os

        data = HeteroData()

        category_indices = []
        relationship_edges = {}

        for path in self.raw_paths:
            with open(path, 'r') as f:
                product = json.load(f)
                category_indices.append(self.categories.index(product['category']))
                for interaction in product["ecology"]:
                    # some ecology items are not GVCS machines or have other relationship types we don't want included
                    rt = interaction['relationship']
                    if rt in self.relationships:
                        if rt not in relationship_edges:
                            relationship_edges[rt] = []
                        other = interaction['tool']
                        if other in self.machines:
                            relationship_edges[rt].append((product['machine'], other))

        data['machine'].x = [category_indices, 1]
        
        for rt in self.relationships: 
            src = [self.machines.index(pair[0]) for pair in relationship_edges[rt]]
            dst = [self.machines.index(pair[1]) for pair in relationship_edges[rt]]
            data['machine', rt, 'machine'].edge_index = torch.tensor([src, dst])

        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])