import os
from absl import logging
from typing import Callable, Dict, Optional, Sequence, Tuple
import gin.torch
import torch
import numpy as np
from .configure_param import get_cvae_param, get_data_loader_param
from datasets.data_loader_lib import DataLoader
from datasets.data_io_lib import IMAGE_KEY, ITEM_NAME_KEY
from models.model_lib import ConditionalVAE, DISCRETE_ENCODER, GAUSSIAN_ENCODER


def argmax_post_processing(predictions: Sequence[torch.Tensor]) -> np.ndarray:
  """Apply argmax to the channel dimension.
  
  Args:
    predictions: A list of prediction tensors of shape (B, C, H, W).
    
  Returns:
    The processed predictions, stacked along the first dimension, hence of shape
    (num_sample, B, C, H, W)
  """
  predictions = torch.stack(predictions, dim=0)
  return predictions.argmax(dim=2).cpu().numpy()


def sigmoid_post_processing(predictions: Sequence[torch.Tensor],
                            threshold: float = 0.5) -> np.ndarray:
  predictions = torch.nn.functional.sigmoid(torch.stack(predictions, dim=0))
  predictions = predictions > threshold
  return predictions.cpu().numpy()


def probabilities_post_processing(
    probabilities: Sequence[torch.Tensor]) -> np.ndarray:
  """Stacking the sample probabilities along the first dimension."""
  return torch.stack(probabilities, dim=0)


def density_post_processing(densities: Sequence[torch.Tensor],
                            scaling: float = 10**5) -> np.ndarray:
  """Stacking the sample densities along the first dimension.
  
  Apply a scaling because the individual densities might be too small. Rarely
  should density be used anyway.
  """
  return torch.stack(densities, dim=0) * scaling


def get_final_processing_layer(
    dataset_name: str, model_name: str
) -> Tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
  if dataset_name == "LIDC_IDRI":
    predictions_processing_layer = sigmoid_post_processing

  if model_name == GAUSSIAN_ENCODER:
    probabilities_processing_layer = density_post_processing
  elif model_name == DISCRETE_ENCODER:
    probabilities_processing_layer = probabilities_post_processing

  return predictions_processing_layer, probabilities_processing_layer


def save_results(base_save_path: str, model_name: str, dataset_name: str,
                 item_name: str, num_sample: int,
                 predictions: Sequence[torch.Tensor],
                 probabilities: Sequence[torch.Tensor]):
  logging.info("saving results for %s model, %s, item %s" %
               (model_name, dataset_name, item_name))

  predictions_processing_layer, probabilities_processing_layer = (
      get_final_processing_layer(dataset_name=dataset_name,
                                 model_name=model_name))
  dataset_model_token = "%s_%s" % (dataset_name, model_name)
  np.save(
      os.path.join(base_save_path, "eval", dataset_model_token, "images",
                   "%s_%dsamples.npy" % (item_name, num_sample)),
      predictions_processing_layer(predictions))
  np.save(
      os.path.join(base_save_path, "eval", dataset_model_token, "quantities",
                   "%s_%dprobs.npy" % (item_name, num_sample)),
      probabilities_processing_layer(probabilities))


@gin.configurable
def eval(model: Optional[ConditionalVAE], check_point_path: Optional[str],
         use_random: bool, top_k: Optional[int], num_sample: Optional[int],
         base_save_path: str):
  has_cuda = True if torch.cuda.is_available() else False
  device = torch.device("cuda" if has_cuda else "cpu")
  Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor

  data_loader_param = get_data_loader_param()
  dataset_name = data_loader_param["dataset_name"]
  dataset = DataLoader(**data_loader_param)

  model_param = get_cvae_param()
  model_name = model_param.encoder_calss

  if model is None:
    model = ConditionalVAE(**model_param).to(device)
    checkpoint = torch.load(check_point_path, map_location="cpu")
    model.load_state_dict(checkpoint)

  test_loader = DataLoader(dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True,
                           sampler=None)

  for _, batch in enumerate(test_loader):
    item_name = batch[ITEM_NAME_KEY][0]
    inputs = batch[IMAGE_KEY].to(device).type(Tensor)
    predictions, probabilities = model.inference(inputs=inputs,
                                                 use_random=use_random,
                                                 top_k=top_k,
                                                 num_sample=num_sample)
    save_results(base_save_path=base_save_path,
                 model_name=model_name,
                 dataset_name=dataset_name,
                 item_name=item_name,
                 num_sample=num_sample,
                 predictions=predictions,
                 probabilities=probabilities)
