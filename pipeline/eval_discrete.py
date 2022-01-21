import numpy as np
import gin.torch
from .eval_lib import GeneralizedEnergyDistanceEvaluator
from datasets.data_io_lib import ITEM_NAME_KEY



gin.parse_config_file('pipeline/configs/eval_LIDC_discrete.gin')
gin.parse_config_file('pipeline/configs/discrete.gin')
gin.parse_config_file('pipeline/configs/common.gin')

if __name__ == '__main__':
    evaluator = GeneralizedEnergyDistanceEvaluator()


    for item_index in range(evaluator.data_io.length):
        item_name = evaluator.data_io.get_data(item_index, output_selection_index=0)[ITEM_NAME_KEY]
        evaluator.compute_for_item(item_index, item_name)
    
    evaluate_result = evaluator.compute_overall_energy_distance()
    print('mean', np.mean(evaluate_result), 'std', np.std(evaluate_result))
