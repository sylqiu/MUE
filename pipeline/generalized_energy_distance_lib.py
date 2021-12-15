from typing import Callable, Dict, Optional, Sequence, Union
import numpy as np


def get_energy_distance_components(
    gt_seg_modes: np.ndarray, seg_samples: np.ndarray,
    mask: Optional[np.ndarray], eval_class_ids: Union[int, Sequence[int]],
    compute_metric: Callable[..., np.ndarray]) -> Dict[str, np.ndarray]:
  """Calculates the components for a metric-based generalized energy distance
  given an array holding all ground truths and an array holding all samples.
  
  Args:
    ground_truths: Tensor of shape (num_ground_truths, C, H, W).
    samples: Tensor of shape (num_samples, C', H, W).
    mask: A binary valued array indicating the validity of the ground truth,
      where 1 means valid.
    eval_class_ids: An integer or list of integers specifying the prediction 
      channels to eval. If integer is given, range() is used.
    compute_metric: The function that computes the metric defined in the
      metric_lib.py.

  Returns:
    A dictionary containing the three energy distance components.
  """
  num_modes = gt_seg_modes.shape[0]
  num_samples = seg_samples.shape[0]

  if isinstance(eval_class_ids, int):
    eval_class_ids = list(range(eval_class_ids))

  d_matrix_YS = np.zeros(shape=(num_modes, num_samples, len(eval_class_ids)),
                         dtype=np.float32)
  d_matrix_YY = np.zeros(shape=(num_modes, num_modes, len(eval_class_ids)),
                         dtype=np.float32)
  d_matrix_SS = np.zeros(shape=(num_samples, num_samples, len(eval_class_ids)),
                         dtype=np.float32)

  # iterate all ground-truth modes
  for mode in range(num_modes):

    ##########################################
    #   Calculate d(Y,S) = [1 - IoU(Y,S)],	 #
    #   with S ~ P_pred, Y ~ P_gt  			 #
    ##########################################

    # iterate the samples S
    for i in range(num_samples):
      d_matrix_YS[mode, i] = compute_metric(gt_seg_modes[mode], seg_samples[i],
                                            mask, eval_class_ids)

    ###########################################
    #   Calculate d(Y,Y') = [1 - IoU(Y,Y')],  #
    #   with Y,Y' ~ P_gt  	   				  #
    ###########################################

    # iterate the ground-truth modes Y' while exploiting the pair-wise symmetries for efficiency
    for mode_2 in range(mode, num_modes):
      metric = compute_metric(gt_seg_modes[mode], gt_seg_modes[mode_2], mask,
                              eval_class_ids)
      d_matrix_YY[mode, mode_2] = metric
      d_matrix_YY[mode_2, mode] = metric

  #########################################
  #   Calculate d(S,S') = 1 - IoU(S,S'),  #
  #   with S,S' ~ P_pred        			#
  #########################################

  # iterate all samples S
  for i in range(num_samples):
    # iterate all samples S'
    for j in range(i, num_samples):

      metric = compute_metric(seg_samples[i], seg_samples[j], mask,
                              eval_class_ids)
      d_matrix_SS[i, j] = metric
      d_matrix_SS[j, i] = metric

  return {'YS': d_matrix_YS, 'SS': d_matrix_SS, 'YY': d_matrix_YY}


def calc_energy_distances(
    d_matrices: Dict[str, np.ndarray],
    source_probability_weighted: Optional[np.ndarray] = None,
    target_probability_weighted: Optional[np.ndarray] = None) -> float:
  """Calculate the energy distance for each image based on matrices holding the
  combinatorial distances.
  
  Args:
    d_matrices: A dictionary containing the energy distance components of each
       testing sample, respectively stacked along the first dimension.
    source_probability_weighted: probability vector of shape
      (num_testing_sample, num_samples)
    target_probability_weighted: probability vector of shape
      (num_testing_sample, num_modes)
    
  Returns:
    The energy distance metric on the testing samples.
  """
  # (num_testing_sample, num_modes, num_samples, num_class)
  d_matrices = d_matrices.copy()

  # perform a nanmean over the class axis so as to not factor in classes that are not present in
  # both the ground-truth mode as well as the sampled prediction
  if (target_probability_weighted
      is not None) and (source_probability_weighted is None):

    mode_probs = target_probability_weighted

    mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)  # average over classes
    mean_d_YS = np.mean(
        mean_d_YS, axis=2
    )  # average over source i.e. samples, since no source probability is provided
    mean_d_YS = mean_d_YS * mode_probs
    d_YS = np.sum(mean_d_YS, axis=1)

    mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
    d_SS = np.mean(mean_d_SS, axis=(1, 2))

    mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
    mean_d_YY = (mean_d_YY * mode_probs[:, :, np.newaxis] *
                 mode_probs[:, np.newaxis, :])
    d_YY = np.sum(mean_d_YY, axis=(1, 2))

  elif (target_probability_weighted is None) and (source_probability_weighted
                                                  is not None):
    mode_probs = source_probability_weighted

    mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
    mean_d_YS = np.mean(mean_d_YS, axis=1)  # average over target
    mean_d_YS = mean_d_YS * mode_probs
    d_YS = np.sum(mean_d_YS, axis=1)

    mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
    d_YY = np.mean(mean_d_YY, axis=(1, 2))

    mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
    mean_d_SS = (mean_d_SS * mode_probs[:, :, np.newaxis] *
                 mode_probs[:, np.newaxis, :])
    d_SS = np.sum(mean_d_SS, axis=(1, 2))

  elif (target_probability_weighted
        is not None) and (source_probability_weighted is not None):
    mode_probs_target = target_probability_weighted
    mode_probs_source = source_probability_weighted

    mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
    mean_d_YS = (mean_d_YS * mode_probs_target[:, :, np.newaxis] *
                 mode_probs_source[:, np.newaxis, :])

    d_YS = np.sum(mean_d_YS, axis=[1, 2])

    mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
    mean_d_SS = (mean_d_SS * mode_probs_source[:, :, np.newaxis] *
                 mode_probs_source[:, np.newaxis, :])
    d_SS = np.sum(mean_d_SS, axis=(1, 2))

    mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
    mean_d_YY = (mean_d_YY * mode_probs_target[:, :, np.newaxis] *
                 mode_probs_target[:, np.newaxis, :])
    d_YY = np.sum(mean_d_YY, axis=(1, 2))

  else:
    mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
    d_YS = np.mean(mean_d_YS, axis=(1, 2))

    mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
    d_SS = np.mean(mean_d_SS, axis=(1, 2))

    mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
    d_YY = np.nanmean(mean_d_YY, axis=(1, 2))

  return 2 * d_YS - d_SS - d_YY
