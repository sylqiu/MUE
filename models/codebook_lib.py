from typing import Tuple
from absl import logging
import torch
from torch import nn
import numpy as np


class QuantizeEMA(nn.Module):

  def __init__(self,
               dim: int,
               n_embed: int,
               init_mean: torch.Tensor,
               init_std: torch.Tensor,
               decay: float = 0.99,
               ):
    """ 
    Args:
      dim: The dimension of the code.
      n_embed: Total number of codes in the code book.
      init_mean: The mean of initialization distribution, per channel.
      init_std: The standard deviation of the initialization distribution, per
        channel.
      decay: Exponential moving average decay.
    """
    super().__init__()

    self.dim = dim
    self.init_decay = decay
    self.n_embed = n_embed
    self.eps = 1e-5
    self.register_buffer('decay', torch.ones([n_embed]) * decay)
    
    embed = torch.randn(dim, n_embed) * init_std + init_mean
    # the code book variable
    self.register_buffer('embed', embed)

    # record the number of features that corresponds to each code and their
    # averages.
    self.register_buffer('cluster_size', torch.ones(n_embed))
    self.register_buffer('embed_avg', embed.clone())

    self.usage_summary = {}


  def embed_code(self, embed_id: torch.Tensor):
    return nn.functional.embedding(embed_id, self.embed.transpose(0, 1))

  def reset_usage_summary(self):
    self.usage_summary = {}

  def record_code_usage_for_batch(self, code_indices: torch.Tensor):
    code_indices = code_indices.cpu().numpy()
    code_indices_list = list(code_indices.squeeze())
    for ind in code_indices_list:
      self.usage_summary[ind] = self.usage_summary.get(ind, 0) + 1

  def log_code_usage(self):
    logging.info(' quantization usage summary \n ')
    for key in sorted(self.usage_summary.keys()):
        logging.info('{} : {}, '.format(key, self.usage_summary[key]))
    logging.info(' dictionary size : {}/{}'.format(len(self.usage_summary), self.n_embed))

  def forward(
      self,
      inputs: torch.Tensor,
      training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
      inputs: The feature tensor of shapae (B, C, H, W) where each spatial
        feature will be quantized. C should equal to self.dim.
      training: torch training argument, during training the code book will be 
        updated by exponential moving average.

    Return:
      The quantized feaure (code), of shape (B, C, H, W); the difference between
      feature and quantized feature of shape (B, C, H, W); the index of the code
      in the code book, of shape (B, H, W).
    """
    inputs = inputs.permute(0, 2, 3, 1)
    
    # flattened input has shape (B*H*W, C)
    flatten = inputs.reshape(-1, self.dim)

    # compute distance table
    dist = (flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed +
            self.embed.pow(2).sum(0, keepdim=True))
    _, embed_ind = (-dist).max(1)
    embed_onehot = nn.functional.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
    embed_ind = embed_ind.view(*inputs.shape[:-1])
    quantized = self.embed_code(embed_ind)



    if training:

      self.cluster_size.data.mul_(self.decay).add_(
          embed_onehot.sum(0).mul_(1 - self.decay))
      embed_sum = flatten.transpose(0, 1) @ embed_onehot
      self.embed_avg.data.mul_(self.decay).add_(embed_sum.mul_(1 - self.decay))
      n = self.cluster_size.sum()
      cluster_size = ((self.cluster_size + self.eps) /
                      (n + self.n_embed * self.eps) * n)
      embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
      self.embed.data.copy_(embed_normalized)


    diff = quantized.detach() - inputs
    quantized = inputs + (quantized - inputs).detach()
    quantized = quantized.permute(0, 3, 1, 2)
    diff = diff.permute(0, 3, 1, 2)
    return quantized, diff, embed_ind

