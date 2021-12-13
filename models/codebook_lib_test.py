import unittest
import torch
from . import codebook_lib


dim = 4
n_embed = 10
BATCHSIZE = 2
HEIGHT = 11
WIDTH = 15
INPUT_CHANNELS = 4

class TestCodebookLib(unittest.TestCase):
    def test_quantizeema(self):
        inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
        layer = codebook_lib.QuantizeEMA(dim, n_embed)
        outputs = layer(inputs)

        self.assertSequenceEqual([outputs[0].shape, outputs[1].shape, outputs[2].shape],
                             [(BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH), 
                             (BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH), 
                             (BATCHSIZE, HEIGHT, WIDTH)])

if __name__ == '__main__':
  unittest.main()
