from typing import Any, Dict, Optional, Sequence, Tuple
import torch
from module_lib import UnetEncoder, UnetDecoder
from utils import loss_lib


class GaussianEncoder(torch.nn.Module):
  """A Encoder that outputs the mean and variance statistics of a Gaussian
  distribution for latent space sampling.
  """

  def __init__(self,
               unet_param: Dict[str, Any],
               latent_code_level: int,
               latent_code_dimension: int,
               exponent_factor: float = 0.5):
    """Initialize the encoder model.

    Args:
      unet_param: The initialization parameter for the UnetEncoder.
      latent_code_level: The level of the output feature to use as the latent
        code, the coarsest level is 0.
      latent_code_dimension: The dimension of the latent code.
      exponent_factor: The multiplicative factor for the exponential
        nonlinearity.
    """
    self._encoder = UnetEncoder(**unet_param)
    self._latent_code_level = latent_code_level
    self._latent_stat_regressor = torch.nn.Conv2d(
        unet_param['channels_list'][-1],
        latent_code_dimension * 2,
        kernel_size=(1, 1))
    self._latent_code_dimension = latent_code_dimension
    self._exponent_factor = exponent_factor

  def forward(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
    batch_size = inputs.shape[0]
    coarse_to_fine_features = self._encoder(inputs)
    latent_code_input = torch.mean(
        coarse_to_fine_features[self.latent_code_level],
        dim=(2, 3),
        keepdim=True)
    mu_log_sigma = self._latent_stat_regressor(latent_code_input)

    # Reshape to (B, C)
    mu_log_sigma = mu_log_sigma.view(batch_size, -1)
    self.mu = mu_log_sigma[:, :self._latent_code_dimension]
    self.sigma = torch.exp(self._exponent_factor *
                           mu_log_sigma[:, self._latent_code_dimension:])

    return coarse_to_fine_features

  def get_distribution(self) -> torch.distributions:
    self.distribution = torch.distributions.Independent(torch.distributions.Normal(
        loc=self.mu, scale=self.sigma), reinterpreted_batch_ndims=1)
    return self.distribution
    
  def sample(self, use_random: bool) -> torch.Tensor:
    if use_random:
      return self.distribution.rsample()
    else:
      return self.mu  

  def get_latent_code_dimension(self):
    return self._latent_code_dimension


class LabelCombinationLayer(torch.nn.Module):

  def __init__(self,
               input_channels: int,
               label_channels: int,
               feature_channels: int,
               kernel_size: int = 3):
    self._conv1 = torch.nn.Sequential([
        torch.nn.Conv2d(label_channels, feature_channels, kernel_size),
        torch.nn.ReLU(inplace=True)
    ])
    self._conv2 = torch.nn.Sequential([
        torch.nn.Conv2d(feature_channels + input_channels, feature_channels,
                        kernel_size),
        torch.nn.ReLU(inplace=True)
    ])

  def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    transformed_label = torch.nn.functional.interpolate(self._conv1(label),
                                                        inputs.shape[2:4])
    return self._conv2(torch.cat((transformed_label, inputs), dim=1))


class LatentCombinationLayer(torch.nn.Module):

  def __init__(self,
               latent_code_dimension: int,
               feature_channels: int,
               combine_method: str = "concat"):
    self._combine_method = combine_method
    if combine_method == "concat":
      self._process_layer = torch.nn.Sequential([
          torch.nn.Conv2d(in_channels=(latent_code_dimension +
                                       feature_channels),
                          out_channels=feature_channels,
                          kernel_size=(1, 1)),
          torch.nn.ReLU(inplace=True)
      ])
    elif combine_method == "sum":
      self._process_layer = torch.nn.Sequential([
          torch.nn.Conv2d(in_channels=latent_code_dimension,
                          out_channels=feature_channels,
                          kernel_size=(1, 1)),
          torch.nn.ReLU(inplace=True)
      ])
    else:
      raise NotImplementedError

  def forward(self, latent_code: torch.Tensor,
              feature: torch.Tensor) -> torch.Tensor:
    height, width = feature.shape[2:4]
    if self._combine_method == "concat":
      outputs = self._process_layer(
          torch.cat((torch.tile(latent_code, (1, 1, height, width)), feature),
                    dim=1))
    elif self._combine_method == "sum":
      outputs = feature + self._process_layer(
          torch.tile(latent_code, (1, 1, height, width)))

    return outputs


class ConditionalVAE(torch.nn.Module):

  def __init__(self, encoder_class: str, prior_encoder_param: Dict[str, Any],
               posterior_encoder_param: Dict[str, Any], input_channels: int,
               label_channels: int, latent_code_dimension: int,
               latent_code_incorporation_level: int, combine_method: str,
               decoder_param: Dict[str, Any]):
    self._encoder_class = encoder_class
    if encoder_class == "Gaussian":
      self._prior_encoder = GaussianEncoder(**prior_encoder_param)
      self._posterior_encoder = GaussianEncoder(**posterior_encoder_param)

    self._decoder = UnetDecoder(**decoder_param)

    self._label_combination_layer = LabelCombinationLayer(
        input_channels,
        label_channels,
        feature_channels=posterior_encoder_param["input_channels"])
    self._latent_combination_layer = LatentCombinationLayer(
        latent_code_dimension,
        features_channels=decoder_param[input_channels],
        combine_method=combine_method)

    self._latent_code_incorporation_level = latent_code_incorporation_level

  def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:

    prior_features_list = self._prior_encoder(inputs)
    self.prior_distribution = self._prior_encoder.get_distribution()

    posterior_inputs = self._label_combination_layer(inputs, label)
    _ = self._posterior_encoder(posterior_inputs)
    self.posterior_distribution = self._posterior_encoder.get_distribution()

    posterior_latent_code = self.posterior_sample(use_random=True)

    decoder_inputs = self._latent_combination_layer(
        posterior_latent_code,
        prior_features_list[self._latent_code_incorporation_level])
    prior_features_list[self._latent_code_incorporation_level] = decoder_inputs
    prediction = self._decoder(prior_features_list)[-1]

    return prediction
  
  def posterior_sample(self, use_random: bool) -> torch.Tensor:
    return self._posterior_encoder.sample(use_random)

  def prior_sample(self, use_random: bool) -> torch.Tensor:
    return self._prior_encoder.sample(use_random)

  def compute_kl_divergence(self):
    if self._encoder_class == "Gaussian":
      return loss_lib.gaussian_kl_functional(self.prior_distribution,
                                             self.posterior_distribution)
    if self._encoder_class == "Discrete":
      return loss_lib.discrete_kl_functional(self.prior_distribution,
                                             self.posterior_distribution)
