import unittest
from utils import IntegerQuantizer
from linear import LinearQuantized
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

        x = torch.rand((5,5))
        my_quantizer(x)

    def test_int8_quant(self):
        pass

    def test_no_momentum_quant(self):
        pass

    def test_sample_tensor(self):
        pass

class TestLinearQuantized(unittest.TestCase):

    def test_linear(self):
        in_features = (3, 2)
        out_features = (4)
        x = torch.arange(6)
        l = LinearQuantized(in_features, out_features)
    

if __name__ == '__main__':
    unittest.main()