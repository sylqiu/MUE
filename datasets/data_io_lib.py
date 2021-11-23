from abc import ABC, abstractmethod
from typing import Any, Dict
from os.path import join as path_join
from absl import logging
import torch
from PIL.Image import open as imread
import numpy as np

IMAGE_KEY = "image"
GROUND_TRUTH_KEY = "ground_truth"
MODE_ID = "mode_id"


class DataIO(ABC):

  @abstractmethod
  def get_data(self, input_index: int,
               output_selection_index: int) -> Dict[str, np.ndarray]:
    pass

  @abstractmethod
  def sample_output_selection_index(self):
    pass

  @abstractmethod
  def get_all_ground_truth_modes(self, input_index: int):
    pass


def get_data_io_by_name(dataset_name: str) -> DataIO:
  if dataset_name == "LIDC_IDRI":
    return LIDC_IDRI
  else:
    raise NotImplementedError


class LIDC_IDRI(DataIO):

  def __init__(self, data_path_root: str, split: str):
    file_list = tuple(open(path_join(data_path_root, "%s.txt" % (split)), "r"))
    self.data_list = [id_.rstrip() for id_ in file_list]
    self.length = len(self.data_list)
    self.data_path_root = path_join(data_path_root, split)
    logging.info("%s split contains %d images" % (split, self.length))

  def _get_ground_truth_name_format(self, input_index: int) -> str:
    return path_join(self.data_path_root, "gt",
                     self.data_list[input_index].replace(".png", "_l%d.png"))

  def get_data(self, input_index: int,
               output_selection_index: int) -> Dict[str, Any]:
    image_name = path_join(self.data_path_root, "images",
                           self.data_list[input_index])
    ground_truth_name = self._get_ground_truth_name_format(input_index) % (
        output_selection_index)

    return {
        IMAGE_KEY: imread(image_name),
        GROUND_TRUTH_KEY: imread(ground_truth_name)
    }

  def sample_output_selection_index(self):
    return torch.randint(0, 4, (1,)).item()

  def get_all_ground_truth_modes(self, input_index: int):
    modes_list = []
    ground_truth_name_format = self._get_ground_truth_name_format(input_index)
    for mode_index in range(4):
      modes_list.append(imread(ground_truth_name_format % (mode_index)))

    return modes_list
