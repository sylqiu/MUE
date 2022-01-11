from typing import Any, Dict, Optional, Sequence, Tuple
import gin.torch
import torch
from torch.autograd import Variable


def get_activation_layer(activation: str):
  if activation == "relu":
    return torch.nn.ReLU(inplace=False)
  else:
    raise NotImplementedError


class Conv2DReLUNorm(torch.nn.Module):

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      kernel_size: int,
      activation: Optional[str],
      use_bias: bool = True,
      use_batchnorm: bool = False,
  ):

    super().__init__()
    if activation is None:
      self._activation = None
    else:
      self._activation = get_activation_layer(activation)

    self._bn = None
    if use_batchnorm:
      use_bias = False
      self._bn = torch.nn.BatchNorm2d(output_dim)

    padding = kernel_size // 2
    self._conv = torch.nn.Conv2d(
        input_dim,
        output_dim,
        kernel_size=kernel_size,
        padding=padding,
        bias=use_bias,
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    outputs = self._conv(inputs)
    if self._activation is not None:
      outputs = self._activation(outputs)
    if self._bn is not None:
      outputs = self._bn(outputs)

    return outputs


class DownSample2D(torch.nn.Module):

  def __init__(self, downsample_scale: int, mode: str):
    super().__init__()
    self._downsample_scale = downsample_scale
    self._mode = mode

  def forward(self,
              inputs: torch.Tensor,
              target_shape: Tuple[int, int] = None) -> torch.Tensor:
    if target_shape is None:
      target_shape = (
          inputs.shape[2] // self._downsample_scale,
          inputs.shape[3] // self._downsample_scale,
      )
    outputs = torch.nn.functional.interpolate(inputs,
                                              mode=self._mode,
                                              size=target_shape,
                                              align_corners=True)
    return outputs


class UpSample2D(torch.nn.Module):

  def __init__(self,
               upsample_scale: Optional[int] = None,
               mode: str = "bilinear"):
    super().__init__()
    self._upsample_scale = upsample_scale
    self._mode = mode

  def forward(self,
              inputs: torch.Tensor,
              target_shape: Tuple[int, int] = None) -> torch.Tensor:
    if target_shape is None and self._upsample_scale is not None:
      target_shape = (
          inputs.shape[2] * self._upsample_scale,
          inputs.shape[3] * self._upsample_scale,
      )
    outputs = torch.nn.functional.interpolate(inputs,
                                              mode=self._mode,
                                              size=target_shape,
                                              align_corners=True)
    return outputs


class ResidualBlock(torch.nn.Module):

  def __init__(
      self,
      channels: int,
      kernel_size_list: Sequence[int],
      normalization_config: Dict[str, Any],
  ):
    super().__init__()
    self._conv_layers = torch.nn.ModuleList()
    self._norm_layers = torch.nn.ModuleList()
    use_bias = normalization_config['use_bias']
    use_bn = normalization_config['use_batchnorm']
    if normalization_config['activation'] is None:
      self._activation = None
    else:
      self._activation = get_activation_layer(
          normalization_config['activation'])

    for kernel_size in kernel_size_list:
      padding = kernel_size // 2
      self._conv_layers.append(
          torch.nn.Conv2d(
              channels,
              channels,
              kernel_size,
              padding=padding,
              bias=use_bias,
          ))
      if use_bn:
        self._norm_layers.append(torch.nn.BatchNorm2d(channels))
      else:
        self._norm_layers.append(None)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    outputs = inputs
    for layer, norm in zip(self._conv_layers, self._norm_layers):
      res = layer(outputs)
      if self._activation is not None:
        res = self._activation(res)
      if norm is not None:
        res = norm(res)

      outputs = outputs + res

    return outputs
