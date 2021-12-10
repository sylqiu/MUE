from typing import Dict, Optional, Tuple
import torch


def gaussian_kl_functional(prior_distribution: torch.distribution,
                           posterior_distribution: torch.distribution):
  """Calculate the KL divergence between Gaussian posterior and prior."""
  return torch.distributions.kl_divergence(posterior_distribution,
                                           prior_distribution)


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


def combine_fedility_losses(fidelity_loss_names_dict: Dict[str, float]):
  loss_mapping = {"binary_segmentation_loss": binary_segmentation_loss}

  def loss_fn(prediction: torch.Tensor, ground_truth: torch.Tensor,
              mask: Optional[torch.Tensor]):
    loss = 0.0
    for name in fidelity_loss_names_dict:
      if name in loss_mapping:
        loss += loss_mapping[name](prediction, ground_truth,
                                   mask) * fidelity_loss_names_dict[name]
      else:
        raise NotImplementedError("%s is not defined in loss_lib!" % name)

    return loss

  return loss_fn
