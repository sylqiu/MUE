import gin.torch
from .train_lib import train
from .eval_lib import GeneralizedEnergyDistanceEvaluator
import torch
from pipeline.configure_param import get_dataset_param
from datasets.dataset_lib import Dataset
from datasets.data_io_lib import ITEM_NAME_KEY



gin.parse_config_file('pipeline/configs/train_MNIST_gaussian.gin')
gin.parse_config_file('pipeline/configs/eval_MNIST_gaussian.gin')
gin.parse_config_file('pipeline/configs/gaussian.gin')
gin.parse_config_file('pipeline/configs/common.gin')



if __name__ == '__main__':
    train()
    #evaluator = GeneralizedEnergyDistanceEvaluator()
    #dataset = Dataset(**get_dataset_param("LIDC_IDRI","YOUR_PATH", "split",  None, None, None, False, True, True))
    "implement YOUR_PATH by dataset path root, split by train or test"


    #for itemname in dataset[ITEM_NAME_KEY]:

    #        evaluator.compute_for_item()
    #print(evaluator.compute_overall_energy_distance())






