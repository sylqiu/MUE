import unittest
import torch
import gin

from . import model_lib, module_lib
from pipeline import configure_param

gin.parse_config_file('pipeline/configs/common.gin')
gin.parse_config_file('pipeline/configs/discrete.gin')
gin.parse_config_file('pipeline/configs/gaussian.gin')


BATCHSIZE = 4
HEIGHT = 256
WIDTH = 256
INPUT_CHANNELS = 3

LABEL_CHANNELS = 1
FEATURE_CHANNLES = 2

LATENT_CODE_DIMENSION = 12

OUTPUT_CHANNELS = 2
CODE_BOOK_SIZE = 10
NORMALIZATION_CONFIG = {'use_batchnorm': 1, 'use_bias': 0, 'activation': 'relu'}


class TestModelLib(unittest.TestCase):

  def test_label_combination_layer(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    labels = torch.ones((BATCHSIZE, LABEL_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    layer = model_lib.LabelCombinationLayer(INPUT_CHANNELS, LABEL_CHANNELS,
                                            FEATURE_CHANNLES)
    outputs = layer(inputs, labels)
    self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, FEATURE_CHANNLES, HEIGHT, WIDTH))

  def test_latent_combination_layer(self):
    latent_code = torch.ones((BATCHSIZE, LATENT_CODE_DIMENSION, HEIGHT, WIDTH),
                             dtype=torch.float32)
    feature = torch.ones((BATCHSIZE, FEATURE_CHANNLES, HEIGHT, WIDTH),
                         dtype=torch.float32)
    layer = model_lib.LatentCombinationLayer(LATENT_CODE_DIMENSION,
                                             FEATURE_CHANNLES)
    outputs = layer(latent_code, feature)
    self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, FEATURE_CHANNLES, HEIGHT, WIDTH))

  def test_latent_code_classifier(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    layer = model_lib.LatentCodeClassifier(INPUT_CHANNELS, OUTPUT_CHANNELS,
                                           CODE_BOOK_SIZE, NORMALIZATION_CONFIG)
    outputs = layer(inputs)
    self.assertSequenceEqual(outputs.shape, (BATCHSIZE, CODE_BOOK_SIZE))

  def test_gaussian_encoder(self):
    inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    PARAM = configure_param.get_gaussian_encoder_param(INPUT_CHANNELS)
    UNET_PARAM = PARAM["unet_encoder_param"]
    KERNEL_SIZE_LIST = UNET_PARAM["kernel_size_list"]
    ENCODER_CHANNEL_LIST = UNET_PARAM["channels_list"]
    layer = model_lib.GaussianEncoder(**PARAM)
    outputs = layer(inputs)
    outputs = outputs[::-1]
    k = 0
    h = HEIGHT
    w = WIDTH
    for i in range(len(KERNEL_SIZE_LIST), 2):
      self.assertSequenceEqual(outputs[k].shape,
                               (BATCHSIZE, ENCODER_CHANNEL_LIST[i], h, w))
      k = k + 1
      h = h // 2
      w = w // 2

  def test_discrete_posterior_encoder(self):

    LATENT_CODE_DIMENSIONS = 128
    PARAM = configure_param.get_discrete_posterior_encoder_param(
        LATENT_CODE_DIMENSIONS)
    inputs = torch.ones((BATCHSIZE, LATENT_CODE_DIMENSIONS, HEIGHT, WIDTH),
                        dtype=torch.float32)
    layer = model_lib.DiscretePosteriorEncoder(**PARAM)
    layer.initialize_code_book(0, 1)
    outputs = layer(inputs)
    self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, LATENT_CODE_DIMENSIONS, 1, 1))

  def test_discrete_prior_encoder(self):
    LATENT_CODE_DIMENSIONS = 12
    PARAM = configure_param.get_discrete_prior_encoder_param(
        LATENT_CODE_DIMENSIONS)
    inputs = torch.ones((BATCHSIZE, LATENT_CODE_DIMENSIONS, HEIGHT, WIDTH),
                        dtype=torch.float32)

    layer = model_lib.DiscretePriorEncoder(**PARAM)
    layer.get_code_book(torch.ones((128, 512), dtype=torch.float32))
    outputs = layer(inputs)
    UNET_PARAM = PARAM["unet_encoder_param"]
    KERNEL_SIZE_LIST = UNET_PARAM["kernel_size_list"]
    ENCODER_CHANNEL_LIST = UNET_PARAM["channels_list"]
    outputs = outputs[::-1]
    k = 0
    h = HEIGHT
    w = WIDTH
    for i in range(len(KERNEL_SIZE_LIST), 2):
      self.assertSequenceEqual(outputs[k].shape,
                               (BATCHSIZE, ENCODER_CHANNEL_LIST[i], h, w))
      k = k + 1
      h = h // 2
      w = w // 2

  def test_conditional_vae_gaussian(self):
    gin.parse_config_file('pipeline/configs/gaussian.gin')
    inputs = torch.ones((BATCHSIZE, 1, HEIGHT, WIDTH), 
                    dtype=torch.float32)
    labels = torch.ones((BATCHSIZE, 1, HEIGHT, WIDTH),
                    dtype=torch.float32)                
    PARAM1 = configure_param.get_cvae_param()
    layer1 = model_lib.ConditionalVAE(**PARAM1)
    outputs1 = layer1(inputs, labels)
    self.assertSequenceEqual(outputs1.shape,
                         (BATCHSIZE, 1, HEIGHT, WIDTH))

    infer1 = layer1.inference(inputs)
    self.assertSequenceEqual(infer1[0][0].shape, 
                         (BATCHSIZE, 1, HEIGHT, WIDTH))

  def test_conditional_vae_discrete(self):
    gin.parse_config_file('pipeline/configs/discrete.gin')
    inputs = torch.ones((BATCHSIZE, 1, HEIGHT, WIDTH), 
                    dtype=torch.float32)
    labels = torch.ones((BATCHSIZE, 1, HEIGHT, WIDTH),
                    dtype=torch.float32)                
    PARAM2 = configure_param.get_cvae_param()
    layer2 = model_lib.ConditionalVAE(**PARAM2)
    layer2.preprocess(**{'mean': 0, 'std': 1})
    outputs2 = layer2(inputs, labels)
    self.assertSequenceEqual(outputs2.shape,
                         (BATCHSIZE, 1, HEIGHT, WIDTH))

    infer2 = layer2.inference(inputs)
    self.assertSequenceEqual(infer2[0][0].shape, 
                         (BATCHSIZE, 1, HEIGHT, WIDTH))






if __name__ == '__main__':
  unittest.main()