from typing import Tuple
import torch


def gaussian_kl_functional(prior_distribution: torch.distribution,
                           posterior_distribution: torch.distribution):
  """Calculate the KL divergence between Gaussian posterior and prior.
  """
  return torch.distributions.kl_divergence(posterior_distribution,
                                           prior_distribution)


def discrete_kl_functional(prior_distribution: torch.Tensor,
                           posterior_distribution: Tuple[torch.Tensor, torch.Tensor],
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
  """
  code_reg_loss = posterior_distribution[0].pow(2).mean()
  classification_loss = torch.nn.functional.binary_cross_entropy(
    prior_distribution, posterior_distribution[1]
  )
  
  return code_reg_loss * beta + classification_loss
  
  