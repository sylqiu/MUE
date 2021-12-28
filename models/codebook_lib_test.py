import unittest
import torch
from . import codebook_lib


dim = 128
n_embed = 512
BATCHSIZE = 16
HEIGHT = 1
WIDTH = 1
INPUT_CHANNELS = 128

class TestCodebookLib(unittest.TestCase):
    def test_quantizeema(self):
        inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
        layer = codebook_lib.QuantizeEMA(dim, n_embed, 0, 1)
        outputs = layer(inputs)

        self.assertSequenceEqual([outputs[0].shape, outputs[1].shape, outputs[2].shape],
                             [(BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH), 
                             (BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH), 
                             (BATCHSIZE, HEIGHT, WIDTH)])

if __name__ == '__main__':
  unittest.main()
