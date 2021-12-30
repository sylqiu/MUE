import unittest
from pipeline.configure_param import get_data_loader_param
from .data_loader_lib import DataLoader
from .data_io_lib import IMAGE_KEY, GROUND_TRUTH_KEY, MODE_ID_KEY, ITEM_NAME_KEY
import torch



class TestDataLoaderLib(unittest.TestCase):
    def test_data_loader(self):

      dataset = DataLoader(**get_data_loader_param("LIDC_IDRI", None, None, None, False, True, True))
      "Remind implement the return LIDC's init of get_data_io_by_name in data_io_lib"

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


















if __name__ == '__main__':
   unittest.main()



