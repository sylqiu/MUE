import unittest
import torch
import model_lib
import gin

BATCHSIZE = 2
HEIGHT = 100
WIDTH = 120
INPUT_CHANNELS = 3

LABEL_CHANNELS = 1
FEATURE_CHANNLES = 2

LATENT_CODE_DIMENSION = 12

OUTPUT_CHANNELS = 2
CODE_BOOK_SIZE = 10
NORMALIZATION_CONFIG = {'use_batchnorm':1, 'use_bias':0, 'activation':'relu'}

class TestModelLib(unittest.TestCase):

    def test_label_combination_layer(self):
        inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
        labels = torch.ones((BATCHSIZE, LABEL_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
        layer = model_lib.LabelCombinationLayer(INPUT_CHANNELS, LABEL_CHANNELS, FEATURE_CHANNLES)
        outputs = layer(inputs, labels)
        self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, FEATURE_CHANNLES, HEIGHT, WIDTH))

    def test_latent_combination_layer(self):
        latent_code = torch.ones((BATCHSIZE, LATENT_CODE_DIMENSION, HEIGHT, WIDTH),
                        dtype=torch.float32)
        feature = torch.ones((BATCHSIZE, FEATURE_CHANNLES, HEIGHT, WIDTH),
                        dtype=torch.float32)
        layer = model_lib.LatentCombinationLayer(LATENT_CODE_DIMENSION, FEATURE_CHANNLES)
        outputs = layer(latent_code, feature)
        self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, FEATURE_CHANNLES, HEIGHT, WIDTH))

    def test_latent_code_classifier(self):
        inputs = torch.ones((BATCHSIZE, INPUT_CHANNELS, HEIGHT, WIDTH),
                        dtype=torch.float32)
        layer = model_lib.LatentCodeClassifier(INPUT_CHANNELS, OUTPUT_CHANNELS, 
                             CODE_BOOK_SIZE, NORMALIZATION_CONFIG)
        outputs = layer(inputs)
        self.assertSequenceEqual(outputs.shape,
                             (BATCHSIZE, CODE_BOOK_SIZE))

    







if __name__ == '__main__':
  unittest.main()