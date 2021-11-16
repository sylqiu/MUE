import torch


def gaussian_kl_functional(prior_distribution, posterior_distribution):
  """Calculate the KL divergence between Gaussian posterior and prior.
  """
  return torch.distributions.kl_divergence(posterior_distribution,
                                           prior_distribution)


def discrete_kl_funcitonal(prior_distribution, posterior_distribution):
  """Calculate the KL divergence between deterministic posterior and categorical
  prior. This amounts to a cross-entropy loss.
  """
  pass