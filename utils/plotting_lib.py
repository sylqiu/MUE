import os
import collections
from typing import Any, Dict, Sequence, Tuple
from absl import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def log_scalar_dict(step, scalar_dict):
  log_str = 'step {} '.format(step)
  for key in scalar_dict:
    log_str += '[{}] {} '.format(key, scalar_dict[key])
  logging.info(log_str)


class AverageMeter(object):
  """Computes and stores the average and current value."""

  def __init__(self):
    self.reset()
    self._moving_avg_multiplier = 0.9

  def reset(self):
    self._val = {}
    self._avg = {}
    self._sum = {}
    self._moving_avg = {}
    self._count = 0

  def update(self, val: Dict[str, Any]):
    self._val = val
    self._count += 1
    for key in val:
      self._sum[key] = self._sum.get(key, 0) + val[key]
      self._avg[key] = self._sum[key] / self._count
      self._moving_avg[key] = self._moving_avg.get(
          key, val[key]) * self._moving_avg_multiplier + (
              1.0 - self._moving_avg_multiplier) * val[key]

  def get_average_dict(self) -> Dict[str, Any]:
    return self._avg

  def get_moving_average_dict(self) -> Dict[str, Any]:
    return self._moving_avg


class FrequencySummarizer():
  """Record the frequency of events described by the coordinates."""

  def __init__(self, table_size: Sequence[int], axis_names: Sequence[str]):
    """Initialize a table, table_size cannot be too large."""
    self.table = np.zeros(table_size)
    self.table_size = table_size
    self.axis_names = axis_names

  def count(self, index):
    self.table[tuple(index)] += 1

  def _normalize(self, table: np.ndarray):
    return table / np.sum(table)

  def retain_axis(self, axes: Sequence[int]):
    """Reduce by summing over the complement of the provided axes."""
    one_hot = [
        True if axis in axes else False for axis in range(self.table_size)
    ]
    sum_axis = np.array(range(len(self.table_size)))[one_hot]
    return np.sum(self.table, tuple(sum_axis))

  def visualize_two_axis(self, axis1: int, axis2: int, cmap: str = 'viridis'):
    new_table = self._normalize(self.retain_axis([axis1, axis2]))
    cm = plt.get_cmap(cmap)
    new_table_img = Image.fromarray(np.array(cm(new_table)[..., :3]))
    new_table_img = new_table_img.resize(
        (self.table_size[axis1] * 2, self.table_size[axis2] * 2), Image.NEAREST)
    visualize_axis_names = [self.axis_names[axis1], self.axis_names[axis2]]
    return new_table, new_table_img, visualize_axis_names


class CreateResultSaver(object):
  """Save validation or test results into numpy files or images."""

  def __init__(self, name, base_dir, token, cmap='viridis'):

    self.name = name
    self.parent_path = os.path.join(base_dir, name, token)
    self.image_path = os.path.join(base_dir, name, token, 'images')
    self.quant_path = os.path.join(base_dir, name, token, 'quantities')
    os.makedirs(self.parent_path, exist_ok=True)
    os.makedirs(self.image_path, exist_ok=True)
    os.makedirs(self.quant_path, exist_ok=True)
    self.scalar_dict = collections.defaultdict(list)
    self.tensor_dict = collections.defaultdict(list)
    self.cmap = plt.get_cmap(cmap)

  def write_image(self, step, image_dict):
    img_save = []
    dict_len = len(image_dict)
    for key, value in image_dict.items():
      img_array = []
      value = value.numpy()
      bdim = value.shape[0]
      for i in range(bdim):
        img = value[i]
        if img.shape[-1] == 1:
          img = img[..., 0]
          img = self.cmap(img)[..., :3]  # only get the RGB
        elif img.shape[-1] == 3:
          img = img
        else:
          raise NotImplementedError

        # pad the image only when there are mutiple images to save
        if dict_len > 1 or bdim > 1:
          img = np.pad(img, ((3, 3), (3, 3), (0, 0)))
        # print(img.shape)
        img_array.append(img)
      img_array = np.concatenate(img_array, axis=1)
      img_save.append(img_array)

    img_save = np.concatenate(img_save, axis=0)
    plt.imsave(os.path.join(self.image_path, '{}.png'.format(step)), img_save)

  def append(self, scalar_dict=None, tensor_dict=None):
    if scalar_dict is not None:
      for key, value in scalar_dict.items():
        self.scalar_dict[key].append(value.numpy())
    if tensor_dict is not None:
      for key, value in tensor_dict.items():
        self.tensor_dict[key].append(value.numpy())

  def save_dict_to_numpy(self):
    if bool(self.scalar_dict):
      for key, value in self.scalar_dict:
        np.save(os.path.join(self.quant_path, key + '_scalar.npy'),
                np.stack(value))
    if bool(self.tensor_dict):
      for key, value in self.tensor_dict:
        np.save(os.path.join(self.quant_path, key + '_tensor.npy'),
                np.stack(value))
