import unittest
from ..pipeline.configure_param import  get_data_loader_param
from .data_loader_lib import DataLoader
from .data_io_lib import MASK_KEY, get_data_io_by_name
from .data_io_lib import IMAGE_KEY, GROUND_TRUTH_KEY, MASK_KEY
import torch

dataset = DataLoader(**get_data_loader_param())

train_loader = DataLoader(dataset,
                            batch_size=10,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            sampler=None)


class TestDataLoaderLib(unittest.TestCase):
    def test_data_loader(self):

      dataset = DataLoader(**get_data_loader_param())

      train_loader = torch.utils.data.DataLoader(dataset,
                            batch_size=10,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            sampler=None)
      for batch in train_loader:
         self.assertSequenceEqual(batch.shape,
                             (10, )










if __name__ == '__main__':
  unittest.main()

