import unittest
import torch

from . import module_lib

BATCHSIZE = 2
HEIGHT = 100
WIDTH = 120
INPUT_CHANNELS = 3
ENCODER_CHANNEL_LIST = [2, 3, 4, 3]
KERNEL_SIZE_LIST = [3, 3, 3, 3]
DOWNSAMPLE_LIST = [0, 1, 0, 1]
ENCODER_OUTPUT_LEVEL_LIST = [1, 1, 1, 1, 1, 1]
NORMALIZATION_CONFIG = {"activation": 'relu'}
DECODER_CHANNEL_LIST = [3, 2, 4, 2]
SKIP_CHANNEL_LIST = [None, 3, 3, None]
DECODER_OUTPUT_LEVEL_LIST = [1, 1, 1, 1]
UPSAMPLE_SCALE = 2
KERNEL_SIZE_LIST = [3, 3]


class TestModuleLib(unittest.TestCase):

  def test_unet_encoder(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    module = module_lib.UnetEncoder(INPUT_CHANNELS, ENCODER_CHANNEL_LIST,
                                    KERNEL_SIZE_LIST, DOWNSAMPLE_LIST,
                                    ENCODER_OUTPUT_LEVEL_LIST,
                                    NORMALIZATION_CONFIG)
    outputs = module(inputs)
    outputs = outputs[::-1]
    k = 0
    h = HEIGHT
    w = WIDTH
    for i in range(len(KERNEL_SIZE_LIST)):
      self.assertSequenceEqual(outputs[i + k].shape,
                               (BATCHSIZE, ENCODER_CHANNEL_LIST[i], h, w))
      if DOWNSAMPLE_LIST[i]:
        k = k + 1
        h = h // 2
        w = w // 2
        self.assertSequenceEqual(outputs[i + k].shape,
                                 (BATCHSIZE, ENCODER_CHANNEL_LIST[i], h, w))

    def test_unet_decoder(self):
      INPUT = module_lib.UnetEncoder(INPUT_CHANNELS, ENCODER_CHANNEL_LIST,
                                     KERNEL_SIZE_LIST, DOWNSAMPLE_LIST,
                                     ENCODER_OUTPUT_LEVEL_LIST,
                                     NORMALIZATION_CONFIG)
      inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                          dtype=torch.float32)
      inputs = INPUT(inputs)
      inputs = [inputs[0], inputs[1], inputs[4]]
      module = module_lib.UnetDecoder(inputs[0].shape[1], DECODER_CHANNEL_LIST,
                                      KERNEL_SIZE_LIST, SKIP_CHANNEL_LIST,
                                      DECODER_OUTPUT_LEVEL_LIST,
                                      NORMALIZATION_CONFIG)
      outputs = module(inputs)

      h = HEIGHT // 4
      w = WIDTH // 4

      for i in range(len(DECODER_OUTPUT_LEVEL_LIST)):

        if SKIP_CHANNEL_LIST[i] is not None:
          h = h * 2
          w = w * 2
          self.assertSequenceEqual(outputs[i].shape,
                                   (BATCHSIZE, DECODER_CHANNEL_LIST[i], h, w))
        else:
          self.assertSequenceEqual(outputs[i].shape,
                                   (BATCHSIZE, DECODER_CHANNEL_LIST[i], h, w))


if __name__ == '__main__':
  unittest.main()