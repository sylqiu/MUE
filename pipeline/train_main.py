import gin.torch
from .train_lib import train



gin.parse_config_file('pipeline/configs/train_discrete.gin')
gin.parse_config_file('pipeline/configs/discrete.gin')
gin.parse_config_file('pipeline/configs/common.gin')


if __name__ == '__main__':
    train()






