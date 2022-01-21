import gin.torch
from .train_lib import train
from absl import logging

logging.set_verbosity(logging.INFO)
gin.parse_config_file('pipeline/configs/train_LIDC_gaussian.gin')
gin.parse_config_file('pipeline/configs/gaussian.gin')



if __name__ == '__main__':

    train()

