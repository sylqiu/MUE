from typing import Optional, Tuple
import gin
import sys
sys.path.append('C:/Users/gli/OneDrive - The Chinese University of Hong Kong/Code/MUE/models')
import model_lib
#from ..models.model_lib import GAUSSIAN_ENCODER, DISCRETE_ENCODER




@gin.configurable
def get_normalization_config(activation: Optional[str], use_bias: bool,
                             use_batchnorm: bool):
  return {
      "activation": activation,
      "use_bias": use_bias,
      "use_batchnorm": use_batchnorm,
  }

@gin.configurable
def get_latent_classifier_param(
    output_channel: int,
    activation: Optional[str], 
    use_bias: bool,
    use_batchnorm: bool

):
  return {
      "output_channel": output_channel,
      "normalization_config": get_normalization_config(
          activation, use_bias, use_batchnorm)
  }


@gin.configurable
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

  for level_index in range(num_level):
    current_channels = min(channels_multiplier * 2**(level_index), max_channels)
    channels_list.extend([current_channels, current_channels])
    kernel_size_list.extend([3, 3])
    downsample_list.extend([True, False])
    output_level_list.extend([False, False, True])

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
  input_channels = min(channels_multiplier * 2**(num_level), max_channels)
  skip_channels_list = [None]
  channels_list = [input_channels]
  kernel_size_list = [3]
  output_level_list = [True]

  for level_index in range(num_level - 2, -1, -1):
    skip_channels = min(channels_multiplier * 2**(level_index), max_channels)
    current_output_channels = min(channels_multiplier * 2**(level_index),
                                max_channels)
    skip_channels_list.extend([skip_channels, None])
    channels_list.extend([skip_channels*2, current_output_channels])
    kernel_size_list.extend([3,3])
    output_level_list.extend([False, True])

    if level_index == 0:
      current_output_channels = output_channels
      skip_channels_list.extend([skip_channels, None])
      channels_list.extend([skip_channels, current_output_channels])
      kernel_size_list.extend([3,3])
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
  if encoder_class == "Gaussian":
    prior_encoder_param = get_gaussian_encoder_param(
        input_channels=input_channels)
    posterior_encoder_param = get_gaussian_encoder_param(
        input_channels=label_combination_layer_output_channels)
  elif encoder_class == "Discrete":
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
      "decoder_param": get_unet_decoder_param(output_channels=label_channels)
  }


@gin.configurable
def get_data_loader_param(
    dataset_name: str, random_crop_size: Optional[Tuple[int, int]],
    random_height_width_ratio_range: Optional[Tuple[float, float]],
    random_rotate_angle_range: Optional[Tuple[float, float]],
    use_random_flip: bool, is_training: bool):
  return {
      "dataset_name": dataset_name,
      "random_crop_size": random_crop_size,
      "random_height_width_ratio_range": random_height_width_ratio_range,
      "random_rotat,e_angle_range": random_rotate_angle_range,
      "use_random_flip": use_random_flip,
      "is_training": is_training
  }

