import unittest
from pipeline.configure_param import get_dataset_param
from .dataset_lib import Dataset
from .data_io_lib import IMAGE_KEY, GROUND_TRUTH_KEY, MODE_ID_KEY, ITEM_NAME_KEY
import torch



class TestDatasetLib_LIDC(unittest.TestCase):
    def test_data_loader(self):

      dataset = Dataset(**get_dataset_param("LIDC_IDRI","YOUR_PATH", "split",  None, None, None, False, True, True))
      "implement YOUR_PATH by dataset path root, split by train or test"

      train_loader = torch.utils.data.DataLoader(dataset,
                            batch_size=10,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            sampler=None,
                            drop_last=True)
                            
      for batch in train_loader:
         self.assertSequenceEqual((batch[IMAGE_KEY].shape, batch[MODE_ID_KEY].shape, 
                             batch[GROUND_TRUTH_KEY].shape),
                             ((10, 1, 180, 180), torch.Size([10]), (10, 1, 180, 180)))

         for itemname in batch[ITEM_NAME_KEY]:
           self.assertSequenceEqual([type(itemname)], [str])
           self.assertSequenceEqual(itemname[0:9], 'LIDC-IDRI')


class TestDatasetLib_MNIST(unittest.TestCase):
    def test_data_loader(self):

      dataset = Dataset(**get_dataset_param("GuessMNIST","YOUR_PATH", "split",  None, None, None, False, True, True))
      "implement YOUR_PATH by dataset path root, split by train or test"

      train_loader = torch.utils.data.DataLoader(dataset,
                            batch_size=10,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            sampler=None,
                            drop_last=True)
                            
      for batch in train_loader:
         self.assertSequenceEqual((batch[IMAGE_KEY].shape, batch[MODE_ID_KEY].shape, 
                             batch[GROUND_TRUTH_KEY].shape),
                             ((10, 1, 28, 112), torch.Size([10]), (10, 1, 28, 112)))

         for itemname in batch[ITEM_NAME_KEY]:
           self.assertSequenceEqual([type(itemname)], [str])




if __name__ == '__main__':
   unittest.main()



