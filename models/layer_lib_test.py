import unittest
import torch
import layer_lib

HAS_CUDA = torch.cuda.is_available()
BATCHSIZE = 2
HEIGHT = 10
WIDTH = 20
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 12


class TestLayerLib(unittest.TestCase):

  def test_conv2d_relu_norm(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    layer = layer_lib.Conv2DReLUNorm(INPUT_CHANNELS, OUTPUT_CHANNELS, 3, 'relu')
    outputs = layer(inputs)

    self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, OUTPUT_CHANNELS, HEIGHT, WIDTH))


if __name__ == '__main__':
  unittest.main()
