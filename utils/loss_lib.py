from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import torch


def gaussian_kl_functional(prior_distribution: torch.distributions,
                           posterior_distribution: torch.distributions):
  """Calculate the KL divergence between Gaussian posterior and prior."""
  return torch.distributions.kl_divergence(posterior_distribution,
                                           prior_distribution).mean()


def discrete_kl_functional(
    prior_distribution: torch.Tensor,
    posterior_distribution: Tuple[torch.Tensor, torch.Tensor],
):
  """Calculate the KL divergence between deterministic posterior and categorical
  prior. This amounts to a cross-entropy loss.
  
  Args:
    prior_distribution: The probability vector on the code book, of shape
      (B, code_book_size).
    posterior_distribution: The difference between quantized and unquantized
      code of shape (B, C), and the code_index of shape (B,).
    
  Returns:
    The loss scalar.
  """

  classification_loss = torch.nn.functional.cross_entropy(
      prior_distribution, posterior_distribution[1])

  return classification_loss


def binary_segmentation_loss(prediction: torch.Tensor,
                             ground_truth: torch.Tensor,
                             mask: Optional[torch.Tensor],
                             gamma: Optional[float] = None) -> torch.Tensor:
  """Compute the binary cross entropy loss with optional focal strength.
  
  Focal loss: https://arxiv.org/abs/1708.02002.
  Args:
    prediction: A tensor of shape (B, 1, H, W).
    ground_truth: A tensor of shape (B, 1, H, W).
    gamma: The higher gamma is, the less weight will be put on the place with
      higher probability.
      
  Returns:
    The loss scalar.
  """
  log_p = -torch.nn.functional.binary_cross_entropy_with_logits(
      prediction, ground_truth, reduction="none")

  if gamma is not None:
    loss = -((1 - torch.exp(log_p) + 1e-2)**gamma) * log_p
  else:
    loss = -log_p

  if mask is None:
    return loss.mean()
  else:
    return torch.divide(loss * mask, mask.sum())


def combine_fidelity_losses(fidelity_loss_config_dict: Dict[str, float]):
  loss_mapping = {"binary_segmentation_loss": binary_segmentation_loss}

  def loss_fn(prediction: torch.Tensor, ground_truth: torch.Tensor,
              mask: Optional[torch.Tensor]):
    loss = 0.0
    for name in fidelity_loss_config_dict:
      if name in loss_mapping:
        loss = loss + loss_mapping[name](prediction, ground_truth,
                                   mask) * fidelity_loss_config_dict[name]
      else:
        raise NotImplementedError("%s is not defined in loss_lib!" % name)

    return loss

  return loss_fn


def combine_loss(loss_dict: Dict[str, torch.Tensor],
                 loss_weight_config: Dict[str, float]) -> torch.Tensor:
  loss = 0.0
  for key in loss_dict:
    loss = loss + loss_dict[key] * loss_weight_config[key]
  
  return loss


def get_current_loss_config(
    current_epoch: int,
    loss_weight_config_list: Sequence[List[Union[int, Dict[str, float]]]]):
  loss_weight_config_list[-1][0] = math.inf
  current_config_index = min(loss_weight_config_list,
                             key=lambda x: x[0] < current_epoch)
  return current_config_index[1]
