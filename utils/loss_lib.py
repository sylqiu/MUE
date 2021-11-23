from typing import Optional, Tuple
import torch


def gaussian_kl_functional(prior_distribution: torch.distribution,
                           posterior_distribution: torch.distribution):
  """Calculate the KL divergence between Gaussian posterior and prior."""
  return torch.distributions.kl_divergence(posterior_distribution,
                                           prior_distribution)


def discrete_kl_functional(prior_distribution: torch.Tensor,
                           posterior_distribution: Tuple[torch.Tensor,
                                                         torch.Tensor],
                           beta: float = 0.1):
  """Calculate the KL divergence between deterministic posterior and categorical
  prior. This amounts to a cross-entropy loss.
  
  We include the code regularization loss here for cleaner implementation.
  
  Args:
    prior_distribution: The probability vector on the code book, of shape
      (B, code_book_size).
    posterior_distribution: The difference between quantized and unquantized
      code of shape (B, C), and the code_index of shape (B,).
    beta: The strength of code regularization loss applied of code difference.
    
  Returns:
    The loss scalar.
  """
  code_reg_loss = posterior_distribution[0].pow(2).mean()
  classification_loss = torch.nn.functional.cross_entropy(
      prior_distribution, posterior_distribution[1])

  return code_reg_loss * beta + classification_loss


def binary_segmentation_loss(prediction: torch.Tensor,
                             ground_truth: torch.Tensor,
                             gamma: Optional[float]) -> torch.Tensor:
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

  return loss.mean()
