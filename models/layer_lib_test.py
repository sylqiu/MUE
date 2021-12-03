import unittest
import torch
import layer_lib

HAS_CUDA = torch.cuda.is_available()
BATCHSIZE = 2
HEIGHT = 10
WIDTH = 20
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 12
DOWNSAMPLE_SCALE = 2
UPSAMPLE_SCALE = 2
KERNEL_SIZE_LIST = [3, 3]


class TestLayerLib(unittest.TestCase):

  def test_conv2d_relu_norm(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    layer = layer_lib.Conv2DReLUNorm(INPUT_CHANNELS, OUTPUT_CHANNELS, 3, 'relu')
    outputs = layer(inputs)

    self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, OUTPUT_CHANNELS, HEIGHT, WIDTH))

  def test_downsample_2d(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    layer = layer_lib.DownSample2D(DOWNSAMPLE_SCALE, 'bilinear')
    outputs = layer(inputs)

    self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, INPUT_CHANNELS, HEIGHT //
                              DOWNSAMPLE_SCALE, WIDTH // DOWNSAMPLE_SCALE))

  def test_upsample_2d(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    layer = layer_lib.UpSample2D(UPSAMPLE_SCALE)
    outputs = layer(inputs)

    self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, INPUT_CHANNELS,
                              HEIGHT * UPSAMPLE_SCALE, WIDTH * UPSAMPLE_SCALE))

  def test_residual_block(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    layer = layer_lib.ResidualBlock(INPUT_CHANNELS, KERNEL_SIZE_LIST, {
        'use_batchnorm': 1,
        'use_bias': 0,
        'activation': 'relu'
    })
    outputs = layer(inputs)

    self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH))


if __name__ == '__main__':
  unittest.main()
