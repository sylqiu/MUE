from typing import Any, Dict, Optional, Sequence
import gin
from ..models.model_lib import GAUSSIAN_ENCODER, DISCRETE_ENCODER


@gin.configurable
def get_normalization_config(activation: Optional[str], use_bias: bool,
                             use_batchnorm: bool):
  return {
      "activation": activation,
      "use_bias": use_bias,
      "use_batchnorm": use_batchnorm,
  }


@gin.configurabale
def get_unet_encoder_param(
    input_channels: int,
    channels_multiplier: int,
    max_channels: int,
    num_level: int,
):
  channels_list = [channels_multiplier]
  kernel_size_list = [3]
  downsample_list = [False]
  output_level_list = [True]

  for level_index in range(q, num_level):
    current_channels = min(channels_multiplier * 2**(level_index), max_channels)
    channels_list.extend([current_channels, current_channels])
    kernel_size_list.extend([3, 3])
    downsample_list.extend([True, False])
    output_level_list.extend([False, True])

  return {
      "input_channels": input_channels,
      "channels_list": channels_list,
      "kernel_size_list": kernel_size_list,
      "downsample_list": downsample_list,
      "output_level_list": output_level_list,
      "normalization_config": get_normalization_config(),
  }


@gin.configurable
def get_unet_decoder_param(
    output_channels: int,
    channels_multiplier: int,
    max_channels: int,
    num_level: int,
):
  input_channels = min(channels_multiplier * 2**(num_level - 2), max_channels)
  skip_channels_list = [None]
  channels_list = [input_channels]
  kernel_size_list = [3]
  output_level_list = [True]

  for level_index in range(num_level - 3, -1, -1):
    skip_channels = min(channels_multiplier * 2**(level_index), max_channels)
    if level_index == 0:
      current_output_channels = output_channels
    else:
      current_output_channels = min(channels_multiplier * 2**(level_index - 1),
                                    max_channels)
    skip_channels_list.extend([skip_channels, None])
    channels_list.extend([skip_channels, current_output_channels])
    kernel_size_list.extend([3, 3])
    output_level_list.extend([False, True])

  return {
      "input_channels": input_channels,
      "channels_list": channels_list,
      "kernel_size_list": kernel_size_list,
      "skip_channels_list": skip_channels_list,
      "output_level_list": output_level_list,
      "normalization_config": get_normalization_config(),
  }


@gin.configurable
def get_gaussian_encoder_param(input_channels: int,
                               latent_code_level: int,
                               latent_code_dimension: int,
                               exponent_factor: float = 0.5):
  return {
      "unet_encoder_param":
          get_unet_encoder_param(input_channels=input_channels),
      "latent_code_level":
          latent_code_level,
      "latent_code_dimension":
          latent_code_dimension,
      "exponent_factor":
          exponent_factor
  }


@gin.configurable
def get_discrete_posterior_encoder_param(input_channels: int,
                                         latent_code_level: int,
                                         latent_code_dimension: int,
                                         code_book_size: int):
  return {
      "unet_encoder_param":
          get_unet_encoder_param(input_channels=input_channels),
      "latent_code_level":
          latent_code_level,
      "latent_code_dimension":
          latent_code_dimension,
      "code_book_size":
          code_book_size
  }


@gin.configurable
def get_discrete_prior_encoder_param(input_channels: int,
                                     latent_code_level: int,
                                     latent_code_dimension: int,
                                     code_book_size: int):
  return {
      "unet_encoder_param":
          get_unet_encoder_param(input_channels=input_channels),
      "latent_code_level":
          latent_code_level,
      "latent_code_dimension":
          latent_code_dimension,
      "code_book_size":
          code_book_size
  }


@gin.configurable
def get_cvae_param(
    encoder_class: str,
    input_channels: int,
    label_channels: int,
    label_combination_layer_output_channels: int,
    latent_code_dimension: int,
    latent_code_incorporation_level: int,
    combine_method: str,
):
  if encoder_class == GAUSSIAN_ENCODER:
    prior_encoder_param = get_gaussian_encoder_param(
        input_channels=input_channels)
    posterior_encoder_param = get_gaussian_encoder_param(
        input_channels=label_combination_layer_output_channels)
  elif encoder_class == DISCRETE_ENCODER:
    prior_encoder_param = get_discrete_prior_encoder_param(input_channels)
    posterior_encoder_param = get_discrete_posterior_encoder_param(
        input_channels=label_combination_layer_output_channels)
  else:
    raise NotImplementedError("%s encoder class is not implemented!" %
                              (encoder_class))

  return {
      "encoder_class": encoder_class,
      "prior_encoder_param": prior_encoder_param,
      "posterior_encoder_param": posterior_encoder_param,
      "input_channels": input_channels,
      "label_channels": label_channels,
      "latent_code_dimension": latent_code_dimension,
      "latent_code_incorporation_level": latent_code_incorporation_level,
      "combine_method": combine_method,
      "decoder_param": get_unet_decoder_param()
  }
