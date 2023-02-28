import unittest
from utils import IntegerQuantizer
import torch

class TestQuantization(unittest.TestCase):

    def test_int4_quant(self):
        my_quantizer = IntegerQuantizer(
            qtype="INT4",
            signed=True,
            momentum=0.01,
            use_ste=True,
            symmetric=False,
        )

        x = torch.random()

    def test_int8_quant(self):
        pass

    def test_no_momentum_quant(self):
        pass

    def test_sample_tensor(self):
        pass

    def test_

if __name__ == '__main__':
    unittest.main()