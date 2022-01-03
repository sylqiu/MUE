import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Union, Tuple
from absl import logging
import torch
from PIL.Image import open as imread
from PIL.Image import Image
import numpy as np
import gin

IMAGE_KEY = "image"
GROUND_TRUTH_KEY = "ground_truth"
MODE_ID_KEY = "mode_id"
MASK_KEY = "mask"
ITEM_NAME_KEY = "item_name"


class DataIO(ABC):

  @abstractmethod
  def get_data(self, input_index: int,
               output_selection_index: int) -> Dict[str, np.ndarray]:
    pass

  @abstractmethod
  def sample_output_selection_index(self):
    pass

  @abstractmethod
  def get_num_groud_truth_modes(self):
    pass

  @abstractmethod
  def get_all_ground_truth_modes(self, item_name: str):
    pass

  @abstractmethod
  def has_ground_truth_modes_probabilities(self):
    pass

  @abstractmethod
  def get_ground_truth_modes_probabilities(self, item_name: str):
    pass



def get_data_io_by_name(dataset_name: str, data_path_root: str,
                        split: str) -> DataIO:
  if dataset_name == "LIDC_IDRI":
    return LIDC_IDRI(data_path_root, split)
  else:
    raise NotImplementedError


class LIDC_IDRI(DataIO):

  def __init__(self, data_path_root: str, split: str):
    file_list = tuple(
        open(os.path.join(data_path_root, "%s.txt" % (split)), "r"))
    self.data_list = [id_.rstrip() for id_ in file_list]
    self.length = len(self.data_list)
    self.data_path_root = os.path.join(data_path_root, split)
    logging.info("%s split contains %d images" % (split, self.length))

  def _get_ground_truth_name_format(self, item_name: str) -> str:
    return os.path.join(self.data_path_root, "gt", item_name + "_l%d.png")

  def get_num_groud_truth_modes(self) -> int:
    return 4

  def get_data(self, input_index: int,
               output_selection_index: int) -> Dict[str, Union[Image, str]]:
    image_name = os.path.join(self.data_path_root, "images",
                              self.data_list[input_index])
    ground_truth_name = self._get_ground_truth_name_format(input_index) % (
        output_selection_index)

    return {
        IMAGE_KEY: imread(image_name),
        GROUND_TRUTH_KEY: imread(ground_truth_name),
        ITEM_NAME_KEY: self.data_list[input_index].replace('.png', '')
    }

  def sample_output_selection_index(self):
    return torch.randint(0, 4, (1,)).item()

  def get_all_ground_truth_modes(self, item_name: int) -> Sequence[Image]:
    modes_list = []
    ground_truth_name_format = self._get_ground_truth_name_format(item_name)
    for mode_index in range(4):
      modes_list.append(imread(ground_truth_name_format % (mode_index)))

    return modes_list

  def has_ground_truth_modes_probabilities(self):
    return False

  def get_ground_truth_modes_probabilities(self, item_name: str):
    return None



@gin.configurable
class GuessMNIST(DataIO):
  """
    Guess the correct digit from an array of 4 digits. The correct digit's label
    follows a certain conditional probability distribution.
  """

  def __init__(self, data_path_root: str, split: str):
    self.digit_images, self.digit_labels = torch.load(
        os.path.join(data_path_root, "%s.pt" % split))
    self.length = self.input_data.shape[0]
    logging.info("%s split contains %d images" % (split, self.length))

    # group data by their label
    self.label_to_indices = {}
    for i in range(self.length):
      self.label_to_indices.setdefault(int(self.digit_labels[i].item()),
                                       []).append(i)

    self.data_path_root = data_path_root

    # data construction rules
    self.modes = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8],
                           [7, 8, 9, 0]])
    self.mode_probabilities = [0.3, 0.2, 0.2, 0.3]
    self.conditional_probabilities = np.array([[0.25, 0.25, 0.25, 0.25],
                                               [0.1, 0.4, 0.1, 0.4],
                                               [0.3, 0.5, 0.1, 0.1],
                                               [0.1, 0.1, 0.1, 0.7]])
    self._cumsum_mode_probs = np.cumsum(self.mode_probabilities)
    self._cumsum_cond_probs = np.cumsum(self.conditional_probabilities, axis=1)

  def _random_sample_image_from_label(
      self, digit_label: int) -> Tuple[np.ndarray, int]:
    label_list = self.label_to_indices[digit_label]
    sample_index = torch.randint(0, len(label_list), (1,)).item()
    return np.array(self.digit_images[label_list[sample_index]]), sample_index

  def _construct_input_output(self, group_index: int,
                              correct_guess_index: int) -> Dict[str, Any]:
    input_image = []
    correct_guess = []
    input_image_indices = []

    for i in range(self.modes.shape[-1]):
      sample_image, sample_image_index = self._random_sample_image_from_label(
          self.modes[group_index][i])
      input_image.append(sample_image)
      input_image_indices.append(sample_image_index)
      if correct_guess_index == i:
        correct_guess.append(sample_image)
      else:
        correct_guess.append(np.zeros_like(sample_image))

    input_image = np.concatenate(input_image, axis=1)
    correct_guess = np.concatenate(correct_guess, axis=1)

    return {
        IMAGE_KEY:
            input_image,
        GROUND_TRUTH_KEY:
            correct_guess,
        ITEM_NAME_KEY:
            '{}_'.format(group_index) +
            ["{}_".format(id) for id in input_image_indices
            ].join("").rstrip("_"),
    }

  def get_num_groud_truth_modes(self) -> int:
    return self.modes.shape[0] * self.modes.shape[1]
  
  def compute_mode_index(self, group_index: int, correct_guess_index: int):
    return group_index * self.modes.shape[-1] + correct_guess_index

  def sample_output_selection_index(self):
    self.group_index = np.argmax(self._cumsum_mode_probs > torch.rand(1).itme())
    self.correct_guess_index = np.argmax(
        self._cumsum_cond_probs[self.group_index] > torch.rand(1).item())
    return self.compute_mode_index(self.group_index, self.correct_guess_index)

  def get_data(
      self, input_index: int,
      output_selection_index: int) -> Dict[str, Union[np.ndarray, str]]:
    # Because data are randomly sampled and generated on the fly,
    # input_index is not used; output_selection_index should be
    # self.group_index * 4 + self.correct_guess_index, which is also not used
    # directly.
    return self._construct_input_output(self.group_index,
                                        self.correct_guess_index)

  def get_all_ground_truth_modes(self, item_name: str) -> Sequence[np.array]:
    img_key = list(map(int, item_name.split('_')))
    group_index = img_key[0]
    input_image_indices = img_key[1:]
    input_labels = self.modes[group_index]
    modes_list = []
    j = 0
    for label, input_image_index in zip(input_labels, input_image_indices):
      output_image = []
      for i in range(self.modes.shape[-1]):
        if i == j:
          output_image.append(
              np.array(self.digit_images[self.label_to_indices[label]
                                         [input_image_index]]))

        else:
          output_image.append(
              np.zeros_like(
                  np.array(self.digit_images[self.label_to_indices[label]
                                             [input_image_index]])))
      output_image = np.concatenate(output_image, axis=1)
      modes_list.append(output_image)

    return modes_list

  def has_ground_truth_modes_probabilities(self):
    return True

  def get_ground_truth_modes_probabilities(self, item_name: str):
    img_key = list(map(int, item_name.split('_')))
    group_index = int(img_key[0])
    return self.conditional_probabilities[group_index]
