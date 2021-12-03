from typing import Any, Dict, Optional, Sequence, Tuple
import torch
from codebook_lib import QuantizeEMA
from layer_lib import get_activation_layer
from models.layer_lib import Conv2DReLUNorm, ResidualBlock
from module_lib import UnetEncoder, UnetDecoder
from ..utils.loss_lib import gaussian_kl_functional, discrete_kl_functional

GAUSSIAN_ENCODER = "Gaussian"
DISCRETE_ENCODER = "Discrete"


class LabelCombinationLayer(torch.nn.Module):
  """The Layer that combines the label and its inputs to feed into the posterior
  encoder.
  
  In this implementation a convolutional layer is first applied to the
  label, then resized to the input shape and concatenated with the input, then
  go through a second convolutional layer.
  """

  def __init__(self,
               input_channels: int,
               label_channels: int,
               feature_channels: int,
               kernel_size: int = 3,
               activation: str = "relu"):
    super().__init__()
    self._conv1 = torch.nn.Sequential([
        torch.nn.Conv2d(label_channels, feature_channels, kernel_size),
        get_activation_layer(activation)
    ])
    self._conv2 = torch.nn.Sequential([
        torch.nn.Conv2d(feature_channels + input_channels, feature_channels,
                        kernel_size),
        get_activation_layer(activation)
    ])

  def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    transformed_label = torch.nn.functional.interpolate(self._conv1(label),
                                                        inputs.shape[2:4])
    return self._conv2(torch.cat((transformed_label, inputs), dim=1))


class LatentCombinationLayer(torch.nn.Module):
  """The Layer that combines the latent code and a feature to feed into the
  decoder.
  
  Supports two method of combination: 'concat' and 'sum'.
  In 'concat' the code is resized to the input shape and concatenated with the
  feature, then they go through a convolutional layer.
  In 'sum' the code goes through a convolutoinal layer whose output channels
  match the feature, then added to the feature.
  """

  def __init__(self,
               latent_code_dimension: int,
               feature_channels: int,
               combine_method: str = "concat",
               activation: str = "relu"):
    """[summary]

    Args:
        latent_code_dimension (int): [description]
        feature_channels (int): [description]
        combine_method (str, optional): [description]. Defaults to "concat".

    Raises:
        NotImplementedError: [description]
    """
    super().__init__()
    self._combine_method = combine_method
    if combine_method == "concat":
      self._process_layer = torch.nn.Sequential([
          torch.nn.Conv2d(in_channels=(latent_code_dimension +
                                       feature_channels),
                          out_channels=feature_channels,
                          kernel_size=(1, 1)),
          get_activation_layer(activation)
      ])
    elif combine_method == "sum":
      self._process_layer = torch.nn.Sequential([
          torch.nn.Conv2d(in_channels=latent_code_dimension,
                          out_channels=feature_channels,
                          kernel_size=(1, 1)),
          get_activation_layer(activation)
      ])
    else:
      raise NotImplementedError

  def forward(self, latent_code: torch.Tensor,
              feature: torch.Tensor) -> torch.Tensor:
    height, width = feature.shape[2:4]
    if len(latent_code.shape) == 2:
      latent_code = torch.tile(latent_code, (1, 1, height, width))
    elif len(latent_code.shape) == 4:
      latent_code = torch.nn.functional.interpolate(latent_code,
                                                    (height, width))
    if self._combine_method == "concat":
      outputs = self._process_layer(torch.cat((latent_code, feature), dim=1))
    elif self._combine_method == "sum":
      outputs = feature + self._process_layer(latent_code)

    return outputs


class LatentCodeClassifier(torch.nn.Module):
  """A soft-max based classier.
  """

  def __init__(self, input_channels: int, output_channels: int,
               code_book_size: int, normalization_config: Dict[str, Any]):
    self._layers = [
        Conv2DReLUNorm(input_channels, output_channels, kernel_size=1)
    ]
    self._layers.append(
        ResidualBlock(output_channels,
                      kernel_size_list=[3, 3, 3],
                      dilation_rate_list=[1, 3, 5],
                      normalization_config=normalization_config))

    self._layers = torch.nn.Sequential(self._layers)
    self._linear_classifier = torch.nn.Linear(output_channels,
                                              code_book_size,
                                              bias=False)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Returns a probability vector of shape (B, code_book_size)."""
    return torch.nn.functional.softmax(self._linear_classifier(
        torch.max(torch.max(self._layers(inputs), dim=-1), dim=-1)),
                                       dim=1)


class GaussianEncoder(torch.nn.Module):
  """A Encoder that outputs the mean and variance statistics of a Gaussian
  distribution for latent space sampling.
  """

  def __init__(self,
               unet_encoder_param: Dict[str, Any],
               latent_code_level: int,
               latent_code_dimension: int,
               exponent_factor: float = 0.5):
    """Initialize the encoder model.

    Args:
      unet_encoder_param: The initialization parameter for the UnetEncoder.
      latent_code_level: The level of the output feature to use as the latent
        code, the coarsest level is 0.
      latent_code_dimension: The dimension of the latent code.
      exponent_factor: The multiplicative factor for the exponential
        nonlinearity, for getting the standard variation statistic.
    """
    super().__init__()
    self._encoder = UnetEncoder(**unet_encoder_param)
    self._latent_code_level = latent_code_level
    self._latent_stat_regressor = torch.nn.Conv2d(
        self._encoder.get_output_channels(),
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
    self.distribution = torch.distributions.Independent(
        torch.distributions.Normal(loc=self.mu, scale=self.sigma),
        reinterpreted_batch_ndims=1)
    return self.distribution

  def sample(self, use_random: bool,
             num_sample: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample num_sample latent codes from the distribution, organized in the
    first dimension.
    
    For the probabilities, we use the density value.
    """
    if use_random:
      # samples should have shape (num_sample, B, latent_code_dimension)
      samples = self.distribution.rsample((num_sample,))
      # densities should have shape (num_sample, B)
      densities = torch.exp(self.distribution.log_prob(samples))
      return samples, densities
    else:
      samples = self.mu.unsqueeze(0)
      densities = torch.exp(self.distribution.log_prob(samples))
      return samples, densities

  def sample_top_k(self, k: int):
    raise ValueError("Gaussian encoder does not support top-k sampling.")

  def get_latent_code_dimension(self):
    return self._latent_code_dimension
  
  def get_input_channels(self):
    return self._encoder.get_input_channels()


class DiscretePosteriorEncoder(torch.nn.Module):
  """A Encoder that learns a code book.
  """

  def __init__(self, unet_encoder_param: Dict[str, Any], latent_code_level: int,
               latent_code_dimension: int, code_book_size: int):
    """Initialize the encoder model.

    Args:
      unet_encoder_param: The initialization parameter for the UnetEncoder.
      latent_code_level: The level of the output feature to use as the latent
        code, the coarsest level is 0.
      latent_code_dimension: The dimension of the latent code.
      code_book_size: The number of latent codes in the code book.
    """
    super().__init__()
    self._encoder = UnetEncoder(**unet_encoder_param)
    self._latent_code_level = latent_code_level
    self._latent_code_dimension = latent_code_dimension
    self._code_book_size = code_book_size

  def initialize_code_book(self, mean, std):
    # self._code_book.embed is of shape (latent_code_dimension, code_book_size)
    self._code_book = QuantizeEMA(self._latent_code_dimension,
                                  self._code_book_size,
                                  init_mean=mean,
                                  init_std=std)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Returns the quantized code of shape (B, C, 1, 1)."""
    unquantized_code = self._encoder(inputs)[self.latent_code_level]
    unquantized_code = torch.mean(unquantized_code, dim=(2, 3), keepdim=True)
    batch_size = inputs.shape[0]
    # Posterior is only used during training
    self.quantized, self.diff, self.code_index = self._code_book(
        unquantized_code, training=True)

    # Squeeze the extra dimensions.
    self.code_index = self.code_index.view((batch_size,))
    self.diff = self.diff.view((batch_size, self._latent_code_dimension))
    return self.quantized

  def get_distribution(self) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.diff, self.code_index

  def sample(self, use_random: bool,
             num_sample: int) -> Tuple[torch.Tensor, Optional[int]]:
    """Get the quantized code, there is no randomness nor multiple sampling."""
    del use_random, num_sample

    # Shape (1, B, latent_code_dimension)
    return self.quantized.unsqueeze(0), None

  def get_latent_code_dimension(self):
    return self._latent_code_dimension


class DiscretePriorEncoder(torch.nn.Module):
  """A Encoder that learns a code book."""

  def __init__(self, unet_encoder_param: Dict[str, Any], latent_code_level: int,
               latent_code_dimension: int, code_book_size: int):
    """Initialize the encoder model.

    Args:
      unet_encoder_param: The initialization parameter for the UnetEncoder.
      latent_code_level: The level of the output feature to use as the latent
        code, the coarsest level is 0.
      latent_code_dimension: The dimension of the latent code.
      code_book_size: The number of latent codes in the code book.
    """
    super().__init__()
    self._encoder = UnetEncoder(**unet_encoder_param)
    self._latent_code_level = latent_code_level
    self._latent_code_dimension = latent_code_dimension
    self._code_book_size = code_book_size
    self._code_classifier = LatentCodeClassifier()
    self._code_book = None

  def get_code_book(self, code_book: torch.Tensor):
    # self._code_book is of shape (latent_code_dimension, code_book_size)
    self._code_book = code_book.detach().clone()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:

    coarse_to_fine_features = self._encoder(inputs)
    self.classification_probability = self._code_classifier(
        coarse_to_fine_features[self._latent_code_level])

    return coarse_to_fine_features

  def get_distribution(self) -> torch.Tensor:
    """Returns a probability vector of shape (B, code_book_size)."""
    return self.classification_probability

  def sample(self, use_random: bool, num_sample: int) -> torch.Tensor:
    """Sample num_sample latent codes from the distribution, organized in the
    first dimension."""
    if use_random:
      # self.classification_probability has shape (batch_size, code_book_size)
      batch_size = self.classification_probability.shape[0]
      val = torch.rand((num_sample, batch_size, 1))
      cutoffs = torch.cumsum(self.classification_probability, dim=1)
      cutoffs = cutoffs.view(1, batch_size, -1)

      # indices should have shape (num_sample, batch_size, 1)
      _, indices = torch.max(cutoffs > val, dim=2)

      # sample should have shape (num_sample, batch_size, latent_code_dimension)
      samples = torch.nn.functional.embedding(indices,
                                              self._code_book.transpose(0, 1))

      classification_probabilities = torch.tile(self.classification_probability,
                                                (num_sample, 1, 1))
      # sample probabilities should have shape
      # (num_sample, batch_size)
      sample_probabilities = torch.gather(classification_probabilities,
                                          dim=2,
                                          index=indices).squeeze(dim=2)
      return samples, sample_probabilities
    else:
      return self.sample_top_k(k=1)

  def sample_top_k(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample the top-k latent codes from the distribution, organized in the
    first dimension."""
    top_probabilities, top_indices = torch.topk(self.classification_probability,
                                                k,
                                                dim=1)
    samples = torch.nn.functional.embedding(top_indices,
                                            self._code_book.transpose(0, 1))

    # top_probabilities should be of shape (k, B)
    top_probabilities = top_probabilities.transpose(0, 1)
    # samples should be of shape (k, B, latent_code_dimension)
    samples = samples.permute((2, 0, 1))
    return samples, top_probabilities

  def get_latent_code_dimension(self):
    return self._latent_code_dimension


class ConditionalVAE(torch.nn.Module):
  """An implementation of generic cVAE."""

  def __init__(self, encoder_class: str, prior_encoder_param: Dict[str, Any],
               posterior_encoder_param: Dict[str, Any], input_channels: int,
               label_channels: int, latent_code_dimension: int,
               latent_code_incorporation_level: int, combine_method: str,
               decoder_param: Dict[str, Any]):
    super().__init__()
    self._encoder_class = encoder_class
    if encoder_class == GAUSSIAN_ENCODER:
      self._prior_encoder = GaussianEncoder(**prior_encoder_param)
      self._posterior_encoder = GaussianEncoder(**posterior_encoder_param)
    elif encoder_class == DISCRETE_ENCODER:
      self._prior_encoder = DiscretePriorEncoder(**prior_encoder_param)
      self._posterior_encoder = DiscretePosteriorEncoder(
          **posterior_encoder_param)
    else:
      raise NotImplementedError("%s encoder class is not implemented!" %
                                (encoder_class))

    self._decoder = UnetDecoder(**decoder_param)

    self._label_combination_layer = LabelCombinationLayer(
        input_channels,
        label_channels,
        feature_channels=self._posterior_encoder.get_input_channels())
    self._latent_combination_layer = LatentCombinationLayer(
        latent_code_dimension,
        features_channels=decoder_param[input_channels],
        combine_method=combine_method)

    self._latent_code_incorporation_level = latent_code_incorporation_level

  def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Returns the prediction tensor, possibly need to be further processed by
    sigmoid or softmax, depending on the applications."""

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
    """Sample one latent codes from the posterior and get rid of the probability
    and the first extra dimension."""
    return self._posterior_encoder.sample(use_random, num_sample=1)[0][0]

  def prior_sample(self, use_random: bool,
                   num_sample: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample num_sample latent codes from the prior, organized in the first
    dimension."""
    return self._prior_encoder.sample(use_random, num_sample)

  def prior_sample_top_k(self, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample the top k latent codes from the prior, organized in the first
    dimension."""
    return self._prior_encdoer.sample_top_k(top_k)

  def compute_kl_divergence(self):
    if self._encoder_class == "Gaussian":
      return gaussian_kl_functional(self.prior_distribution,
                                    self.posterior_distribution)
    if self._encoder_class == "Discrete":
      return discrete_kl_functional(self.prior_distribution,
                                    self.posterior_distribution)

  def inference(
      self,
      inputs: torch.Tensor,
      use_random: bool = True,
      top_k: Optional[int] = None,
      num_sample: int = 1
  ) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    """Returns the predictions and their probabilities."""
    if (self._encoder_class == "Discrete" and
        self._prior_encoder._code_book is None):
      self._prior_encoder.get_code_book(self._posterior_encoder.code_book.embed)

    prior_features_list = self._prior_encoder(inputs)
    latent_combination_feature = prior_features_list[
        self._latent_code_incorporation_level].detach().clone()
    self.prior_distribution = self._prior_encoder.get_distribution()

    # Latent codes and probabilities are organized in the first dimension.
    # We prioritize top-k sampling.
    if top_k is not None:
      latent_codes, probabilities = self.prior_sample_top_k(top_k)
    else:
      latent_codes, probabilities = self.prior_sample(use_random=use_random,
                                                      num_sample=num_sample)

    predictions = []

    for sample_index in range(latent_codes.shape[-1]):
      latent_code = latent_codes[sample_index]
      decoder_inputs = self._latent_combination_layer(
          latent_code, latent_combination_feature)
      prior_features_list[
          self._latent_code_incorporation_level] = decoder_inputs

      predictions.append(self._decoder(prior_features_list)[-1])

    return predictions, torch.split(probabilities,
                                    probabilities.shape[0],
                                    dim=0)
