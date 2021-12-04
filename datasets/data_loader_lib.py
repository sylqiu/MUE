from typing import Any, Dict, Optional, Sequence, Tuple
import gin.torch
import torch
import torchvision.transforms.functional as TF
import numpy as np
from .data_io_lib import get_data_io_by_name, IMAGE_KEY, GROUND_TRUTH_KEY, MODE_ID_KEY

MAX_INT8 = 255.0


def to_numpy(pil_image: Any) -> np.ndarray:
  return np.array(pil_image) / MAX_INT8


def conform_channel_dim(image: np.ndarray):
  if len(image.shape) == 2:
    # add one singleton dimension.
    return image[np.newaxis, ...]
  elif len(image.shape) == 3:
    return image
  else:
    raise ValueError("Image has unsupported rank.")


def package_image_data(pil_image_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
  data_dict = {}
  for key in pil_image_dict:
    data_dict[key] = conform_channel_dim(to_numpy(pil_image_dict[key]))

  return data_dict


def random_crop(data_dict: Dict[str, Any], random_crop_size: Tuple[int, int],
                random_ratio: Tuple[float, float]) -> Dict[str, Any]:
  if random_ratio is None:
    random_ratio = [1.0, 1.0]

  ori_height, ori_width = data_dict[IMAGE_KEY].height, data_dict[
      IMAGE_KEY].width
  hs = torch.randint(int(random_crop_size[0]), int(random_crop_size[1] + 1),
                     (1,)).item()

  random_ratio = torch.rand(1).item() * (-random_ratio[0] +
                                         random_ratio[1]) + random_ratio[0]
  ws = hs * ori_width / ori_height * random_ratio

  if hs > ori_height or ws > ori_width:
    hp = int(np.abs(-ori_height + hs))
    wp = int(np.abs(-ori_width + ws))
    x_p = torch.randint(0, int(wp + 1), (1,)).item()
    y_p = torch.randint(0, int(hp + 1), (1,)).item()
    return {
        IMAGE_KEY:
            TF.resized_crop(TF.pad(data_dict[IMAGE_KEY], (wp, hp)), y_p, x_p,
                            hs, ws, [ori_height, ori_width]),
        GROUND_TRUTH_KEY:
            TF.resized_crop(TF.pad(data_dict[GROUND_TRUTH_KEY], (wp, hp)), y_p,
                            x_p, hs, ws, [ori_height, ori_width])
    }
  else:
    x_p = torch.randint(0, int(ori_width - ws + 1), (1,)).item()
    y_p = torch.randint(0, int(ori_height - hs + 1), (1,)).item()
    return {
        IMAGE_KEY:
            TF.resized_crop(data_dict[IMAGE_KEY], y_p, x_p, hs, ws,
                            [ori_height, ori_width]),
        GROUND_TRUTH_KEY:
            TF.resized_crop(data_dict[GROUND_TRUTH_KEY], y_p, x_p, hs, ws,
                            [ori_height, ori_width])
    }


def random_flip(data_dict: Dict[str, Any]) -> Dict[str, Any]:
  coin = torch.randint(0, 2, (1,)).item()
  if coin == 0:
    return {
        IMAGE_KEY: TF.hflip(data_dict[IMAGE_KEY]),
        GROUND_TRUTH_KEY: TF.hflip(data_dict[GROUND_TRUTH_KEY])
    }
  else:
    return data_dict


def random_rotate(
    data_dict: Dict[str, Any],
    random_rotate_angle_range: Tuple[float, float]) -> Dict[str, Any]:
  angle = torch.randint(int(random_rotate_angle_range[0]),
                        int(random_rotate_angle_range[1] + 1), (1,)).item()
  return {
      IMAGE_KEY:
          TF.rotate(data_dict[IMAGE_KEY], angle, resample=3, fill=(0,)),
      GROUND_TRUTH_KEY:
          TF.rotate(data_dict[GROUND_TRUTH_KEY], angle, resample=3, fill=(0,))
  }


class DataLoader(torch.utils.data.dataset):

  def __init__(self, dataset_name: str, random_crop_size: Optional[Tuple[int,
                                                                         int]],
               random_height_width_ratio_range: Optional[Tuple[float, float]],
               random_rotate_angle_range: Optional[Tuple[float, float]],
               use_random_flip: bool, is_training: bool):
    self._data_io_class = get_data_io_by_name(dataset_name)
    self._random_crop_size = random_crop_size
    self._random_rotate_angle_range = random_rotate_angle_range
    self._use_random_flip = use_random_flip
    self._random_height_width_ratio_range = random_height_width_ratio_range
    self._is_training = is_training

  def __getitem__(self, input_index: int):
    if self._is_training:
      mode_id = self._data_io_class.sample_output_selection_index()
    else:
      mode_id = 0

    data_dict = self._data_io_class.get_data(input_index, mode_id)

    if self._random_crop_size:
      data_dict = random_crop(data_dict, self._random_crop_size,
                              self._random_height_width_ratio_range)
    if self._use_random_flip:
      data_dict = random_flip(data_dict)
    if self._random_rotate_angle_range:
      data_dict = random_rotate(data_dict, self._random_rotate_angle_range)

    data_dict = package_image_data(data_dict)
    data_dict[MODE_ID_KEY] = mode_id

    return data_dict

  def get_all_ground_truth_modes(self,
                                 input_index: int) -> Sequence[np.ndarray]:
    modes_list = []
    for img in self._data_io_class.get_all_ground_truth_modes(input_index):
      modes_list.append(to_numpy(img))

    return modes_list
