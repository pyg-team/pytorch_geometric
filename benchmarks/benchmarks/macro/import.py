from benchmarks import Benchmark

class Import(Benchmark):
    rounds = 1

    def timeraw_import(self):
        return "import torch_geometric"
