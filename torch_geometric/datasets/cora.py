# 2708 vertices
# 5429 edges
# 1433 features
# 7 classes

import os


class Cora(object):
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):
        self.root = os.path.expanduser(root)
        self.processed_folder = os.path.join(self.root, 'processed')
        print(self.root)

        file = os.path.join(self.root, 'cora.content')
        with open(file, 'r') as f:
            for line in f:
                s = line.split('\t')
                idx = s[0]
                features = s[1:-1]
                label = s[-1]
                print(idx)
                print(label)
                print(len(features))
            print(f)

        pass


# Training: 20 samples per class
# Validation and tests sets consists of 500 and 1000 disjoint vertices??
