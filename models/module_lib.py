from typing import Any, Dict, Optional, Sequence
import torch
from .layer_lib import Conv2DReLUNorm, DownSample2D, UpSample2D


class UnetEncoder(torch.nn.Module):

  def __init__(
      self,
      input_channels: int,
      channels_list: Sequence[int],
      kernel_size_list: Sequence[int],
      downsample_list: Sequence[bool],
      output_level_list: Sequence[bool],
      normalization_config: Dict[str, Any],
  ):
    """Initialize the encoder model.

      Args:
        input_channels: The number of channels in the input.
        channels_list: The number of output channels of
          the intermediate layers.
        kernel_size_list: The kernel size of the intermediate
          layers.
        downsample_list: The downsample scale at each intermediate layer.
        output_level_list: An array indicating which levels will be returned in
          the forward pass.
        normalization_config: The normalization configuration for the
          intemediate layers.
    """
    super().__init__()
    self._layers = torch.nn.ModuleList()
    self._input_channels = input_channels
    for layer_index in range(len(kernel_size_list)):
      output_channels = channels_list[layer_index]
      self._layers.append(
          Conv2DReLUNorm(input_channels, output_channels,
                         kernel_size_list[layer_index], **normalization_config))
      if downsample_list[layer_index]:
        self._layers.append(DownSample2D(downsample_scale=2, mode='bilinear'))
      input_channels = output_channels

    self._output_level_list = output_level_list
    self._output_channels = output_channels
    
  def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
    """Returns the features from coarse to fine."""
    return_list = []
    output = x
    for layer_index, layer in enumerate(self._layers):
      
      output = layer(output)
      if self._output_level_list[layer_index]:
        return_list.append(output)


    return return_list[::-1]
  
  def get_input_channels(self):
    return self._input_channels

  def get_output_channels(self):
    return self._output_channels


class UnetDecoder(torch.nn.Module):

  def __init__(
      self,
      input_channels: int,
      channels_list: Sequence[int],
      kernel_size_list: Sequence[int],
      skip_channels_list: Sequence[Optional[int]],
      output_level_list: Sequence[bool],
      normalization_config: Dict[str, Any],
  ):
    super().__init__()
    self._layers = torch.nn.ModuleList()
    self._input_channels = input_channels
    self._skip_channels_list = skip_channels_list
    self._output_level_list = output_level_list

    for layer_index in range(len(kernel_size_list)):
      output_channels = channels_list[layer_index]
      if skip_channels_list[layer_index] is not None:
        input_channels += skip_channels_list[layer_index]
        self._layers.append(UpSample2D(None))
        self._layers.append(
            Conv2DReLUNorm(input_channels, output_channels,
                           kernel_size_list[layer_index],
                           **normalization_config))
      else:
        self._layers.append(
            Conv2DReLUNorm(input_channels, output_channels,
                           kernel_size_list[layer_index],
                           **normalization_config))
      input_channels = output_channels
    self._output_channels = output_channels


  def forward(self, x: Sequence[torch.Tensor]):
    skip_index = 1
    layer_index = 0
    outputs = x[0]
    outputs_list = []
    for level_index in range(len(self._skip_channels_list)):
      if self._skip_channels_list[level_index] is not None:
        upsample_layer = self._layers[layer_index]
        conv_layer = self._layers[layer_index+1]
        layer_index += 2
        skip_inputs = x[skip_index]
        outputs = torch.cat((upsample_layer(outputs, skip_inputs.shape[2:4]), skip_inputs),
                            axis=1)
        outputs = conv_layer(outputs)
        skip_index += 1
      else:
        conv_layer = self._layers[layer_index]
        outputs = conv_layer(outputs)
        layer_index += 1


      if self._output_level_list[level_index]:
        outputs_list.append(outputs)

    return outputs_list
  
  def get_input_channels(self):
    return self._input_channels

  def get_output_channels(self):
    return self._output_channels
    
  
    
    
      
    
