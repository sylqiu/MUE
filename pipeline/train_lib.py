import os
from absl import logging
from typing import Callable, Dict, Optional, Sequence, Tuple
import gin.torch
import torch
from .configure_param import get_cvae_param, get_data_loader_param
from datasets.data_loader_lib import DataLoader
from datasets.data_io_lib import MASK_KEY, get_data_io_by_name
from datasets.data_io_lib import IMAGE_KEY, GROUND_TRUTH_KEY, MASK_KEY
from models.model_lib import ConditionalVAE, DISCRETE_ENCODER, GAUSSIAN_ENCODER
from utils.plotting_lib import AverageMeter, log_scalar_dict
from utils.loss_lib import combine_fedility_losses, combine_loss, get_current_loss_config
from .eval_lib import eval


def train_epoch(model: ConditionalVAE, data_loader: torch.utils.data.DataLoader,
                fidelity_loss_fn: Callable[
                    [torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
                    torch.Tensor], loss_weight_config: Dict[str, float],
                optimizer: torch.optim.Optimizer, average_meter: AverageMeter,
                device: str):

  Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor

  def train_step(batch: Dict[str, torch.Tensor]):
    inputs = batch[IMAGE_KEY].to(device).type(Tensor)
    ground_truth = batch[GROUND_TRUTH_KEY].to(device).type(Tensor)
    mask = None
    if MASK_KEY in batch:
      mask = batch[MASK_KEY].to(device).type(Tensor)

    prediction = model.forward(inputs=inputs, label=ground_truth)

    loss_dict = {}
    loss_dict["kl"] = model.compute_kl_divergence()
    loss_dict["data_fidelity"] = fidelity_loss_fn(prediction, ground_truth,
                                                  mask)
    loss_dict["regularization"] = model.compute_regularization_loss()

    average_meter.update(loss_dict)

    loss = combine_loss(loss_dict, loss_weight_config)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  average_meter.reset()
  for step_index, batch in enumerate(data_loader):
    train_step(batch)

    if step_index % 50 == 0:
      log_scalar_dict(step_index, average_meter.get_moving_average_dict())


def adaptive_code_book_initialization(model: ConditionalVAE,
                                      data_loader: torch.utils.data.DataLoader,
                                      device: torch.device):
  Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
  code_stat = AverageMeter()
  for _, batch in enumerate(data_loader):
    inputs = batch[IMAGE_KEY].to(device).type(Tensor)
    ground_truth = batch[GROUND_TRUTH_KEY].to(device).type(Tensor)
    _ = model.forward(inputs=inputs, label=ground_truth)
    code = model.posterior_sample(use_random=False)

    code_stat.update({
        "variance": code.pow(2).mean(dim=0, keepdim=True),
        "mean": code.mean(dim=0, keepdim=True)
    })

  initial_std = code_stat.get_average_dict()["variance"].pow(0.5).squeeze(3).squeeze(2)
  initial_mean = code_stat.get_average_dict()["mean"].squeeze(3).squeeze(2)

  model.preprocess(mean=initial_mean.transpose(0,1), std=initial_std.transpose(0,1))


@gin.configurable
def train(batch_size: int,
          num_epochs: int,
          base_save_path: str,
          fidelity_loss_config_dict: Dict[str, float],
          loss_weight_config_list: Sequence[Tuple[int, Dict[str, float]]],
          initial_learning_rate: float,
          learning_rate_milestones: Dict[int, float],
          eval_epoch_interval: int = 10,
          eval_top_k_samples: Optional[int] = 8,
          eval_num_samples: Optional[int] = 8):
  """The training function.

  Args:
      batch_size: The training batch size.
      num_epochs: The total number of epochs to train.
      base_save_path: Where the data will be saved.
      fidelity_loss_config_dict: The fidelity losses to use and their weight.
      loss_weight_config_list: A list of (use_up_to_epoch, loss_weight_dict),
        where loss_weight_dict contains the weights for kl, data_fidelity and
        regularization losses.
      initial_learning_rate: The initial learning rate.
      learning_rate_milestones: A dictionary of form {milestone: learning_rate},
        where if epoch > milestone, leanrning_rate will be used.
      eval_epoch_interval: The number of epochs trained between intermediate
        evaluation.
      eval_top_k_sample: If specified and supported by the model, use top-k
        samples.
      eval_num_sample: If specified and supported, use random sampling with
        num_samples.
  """

  has_cuda = True if torch.cuda.is_available() else False
  device = torch.device("cuda" if has_cuda else "cpu")

  data_loader_param = get_data_loader_param()
  dataset_name = data_loader_param["dataset_name"]
  dataset = DataLoader(**data_loader_param)

  model_param = get_cvae_param()
  model_name = model_param["encoder_class"]

  if model_name == GAUSSIAN_ENCODER:
    eval_use_random = True
  elif model_name == DISCRETE_ENCODER:
    eval_use_random = False

  model = ConditionalVAE(**model_param).to(device)
  train_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True,
                                             sampler=None)
  optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
  fidelity_loss_fn = combine_fedility_losses(fidelity_loss_config_dict)
  average_meter = AverageMeter()

  dataset_model_token = "%s_%s" % (dataset_name, model_name)

  # perform adaptive code book initialization for discrete posterior encoder
  if model.get_encoder_class() == DISCRETE_ENCODER:
    model.preprocess(**{'mean': 0, 'std': 1})
    adaptive_code_book_initialization(model, train_loader, device)


  milestone_index = 0
  for epoch_index in range(num_epochs):
    loss_weight_dict = get_current_loss_config(epoch_index, loss_weight_config_list)
    if epoch_index + 1 in learning_rate_milestones:
      logging.info("Using learning rate %f" %
                   (learning_rate_milestones[milestone_index]))
      for pg in optimizer.param_groups:
        pg["lr"] = learning_rate_milestones[milestone_index]

      milestone_index += 1

    train_epoch(model=model,
                data_loader=train_loader,
                fidelity_loss_fn=fidelity_loss_fn,
                loss_weight_config=loss_weight_dict,
                optimizer=optimizer,
                average_meter=average_meter,
                device=device)
    print('gone')

    if (epoch_index + 1) % eval_epoch_interval == 0 or (epoch_index +
                                                        1) == num_epochs:
      torch.save(
          model.state_dict(),
          os.path.join(base_save_path, "train", dataset_model_token,
                       "epoch_%d.pth"))
      eval(model,
           check_point_path=None,
           use_random=eval_use_random,
           top_k=eval_top_k_samples,
           num_sample=eval_num_samples,
           base_save_path=base_save_path,
           epoch_index=epoch_index)
