from typing import Optional, Sequence, Union
import numpy as np


def get_metric_function_by_name(name: str):
  if name == "IOU":
    return compute_iou_metric
  else:
    raise NotImplementedError


def softmax_to_onehot(arr: np.ndarray) -> np.ndarray:
  """Transform a numpy array of softmax values into a one-hot encoded array.
  
  Assumes classes are encoded in axis 1.
  """
  num_classes = arr.shape[1]
  arr_argmax = np.argmax(arr, axis=1)

  for c in range(num_classes):
    arr[:, c] = (arr_argmax == c).astype(np.uint8)
  return arr


def numpy_one_hot(label_arr: np.ndarray, num_classes: int) -> np.ndarray:
  """One-hotify an integer-labeled numpy array. One-hot encoding is encoded in additional last axis.
  """
  # replace labels >= num_classes with 0
  label_arr[label_arr >= num_classes] = 0

  res = np.eye(num_classes)[np.array(label_arr).reshape(-1)]
  return res.reshape(list(label_arr.shape) + [num_classes])


def calc_confusion(
    labels: np.ndarray,
    samples: np.ndarray,
    mask: Optional[np.ndarray],
    class_axis: Union[int, Sequence[int]],
):
  """Compute confusion matrix for each class across the given arrays.
    
  Assumes classes are given in integer-valued encoding.
  
  Args:
  labels: of shape (1, num_class, h, w)
  samples: of shape (1, num_class, h, w)
  mask: A binary valued array indicating the validity of the label, where 1
    means valid.
  class_axis: integer or list of integers specifying the classes to evaluate
  """
  try:
    assert labels.shape == samples.shape
  except:
    raise AssertionError('shape mismatch {} vs. {}'.format(
        labels.shape, samples.shape))

  if isinstance(class_axis, int):
    num_classes = class_axis
    class_axis = range(class_axis)
  elif isinstance(class_axis, list):
    num_classes = len(class_axis)
  else:
    raise TypeError('Arg class_axis needs to be int or list, not {}.'.format(
        type(class_axis)))

  if mask is None:
    shp = labels.shape
    mask = np.ones(shape=(shp[0], 1, shp[2], shp[3]))

  conf_matrix = np.zeros(shape=(num_classes, 4), dtype=np.float32)
  for i, c in enumerate(class_axis):

    pred_ = (samples == c).astype(np.uint8)
    labels_ = (labels == c).astype(np.uint8)

    conf_matrix[i, 0] = int(
        ((pred_ != 0) * (labels_ != 0) * (mask != 0)).sum())  # TP
    conf_matrix[i, 1] = int(
        ((pred_ != 0) * (labels_ == 0) * (mask != 0)).sum())  # FP
    conf_matrix[i, 2] = int(
        ((pred_ == 0) * (labels_ == 0) * (mask != 0)).sum())  # TN
    conf_matrix[i, 3] = int(
        ((pred_ == 0) * (labels_ != 0) * (mask != 0)).sum())  # FN

  return conf_matrix


def metrics_from_conf_matrix(conf_matrix):
  """
    Calculate IoU per class from a confusion_matrix.
    :param conf_matrix: 2D array of shape (num_classes, 4)
    :return: dict holding 1D-vectors of metrics
    """
  tps = conf_matrix[:, 0]
  fps = conf_matrix[:, 1]
  fns = conf_matrix[:, 3]

  metrics = {}
  metrics['iou'] = np.zeros_like(tps, dtype=np.float32)

  # iterate classes
  for c in range(tps.shape[0]):
    # unless both the prediction and the ground-truth is empty, calculate a finite IoU
    if tps[c] + fps[c] + fns[c] != 0:
      metrics['iou'][c] = tps[c] / (tps[c] + fps[c] + fns[c])
    else:
      metrics['iou'][c] = 1
      # metrics['iou'][c] = np.nan

  return metrics


def compute_iou_metric(ground_truth: np.ndarray, sample: np.ndarray,
                       mask: Optional[np.ndarray],
                       eval_class_ids: Union[int, Sequence[int]]):
  conf_matrix = calc_confusion(ground_truth, sample, mask, eval_class_ids)
  return 1.0 - metrics_from_conf_matrix(conf_matrix)['iou']


def compute_l1_metric(ground_truth: np.ndarray, sample: np.ndarray,
                      mask: Optional[np.ndarray],
                      eval_class_ids: Union[int, Sequence[int]]):
  return np.sum(compute_l1_diff(ground_truth, sample, mask))


def compute_l1_diff(ground_truth: np.ndarray, sample: np.ndarray,
                    mask: Optional[np.ndarray]):
  diff = np.abs(ground_truth - sample)
  if mask is not None:
    diff = diff * mask

  return diff
