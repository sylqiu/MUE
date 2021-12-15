from absl import logging
from typing import Callable, Dict, Optional
import torch
from .configure_param import get_cvae_param, get_data_loader_param
from ..datasets.data_loader_lib import DataLoader
from ..datasets.data_io_lib import MASK_KEY, get_data_io_by_name
from ..datasets.data_io_lib import IMAGE_KEY, GROUND_TRUTH_KEY, MASK_KEY
from ..models.model_lib import ConditionalVAE
from ..utils.plotting_lib import AverageMeter, log_scalar_dict
from ..utils.loss_lib import combine_fedility_losses


def combine_loss(loss_dict: Dict[str, torch.Tensor],
                 loss_weight_config: Dict[str, float]) -> torch.Tensor:
  loss = 0.0
  for key in loss_dict:
    loss_dict += loss_dict[key] * loss_weight_config[key]

  return loss


def train_epoch(model: ConditionalVAE, data_loader: torch.utils.data.dataLoader,
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
    loss_dict["data fidelity"] = fidelity_loss_fn(prediction, ground_truth,
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


def train(batch_size: int, num_epochs: int,
          fidelity_loss_names_dict: Dict[str, float],
          cvae_loss_weight_dict: Dict[str, float], initial_learning_rate: float,
          learning_rate_milestones: Dict[int, float]):

  has_cuda = True if torch.cuda.is_available() else False
  device = torch.device("cuda" if has_cuda else "cpu")

  dataset = DataLoader(**get_data_loader_param())
  model = ConditionalVAE(**get_cvae_param()).to(device)
  train_loader = torch.utils.data.DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            sampler=None)
  optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
  fidelity_loss_fn = combine_fedility_losses(fidelity_loss_names_dict)
  average_meter = AverageMeter()

  milestone_index = 0
  for epoch_index in range(num_epochs):
    if epoch_index + 1 in learning_rate_milestones:
      logging.info("Using learning rate %f" %
                   (learning_rate_milestones[milestone_index]))
      for pg in optimizer.param_groups:
        pg["lr"] = learning_rate_milestones[milestone_index]

      milestone_index += 1

    train_epoch(model=model,
                data_loader=train_loader,
                fidelity_loss_fn=fidelity_loss_fn,
                loss_weight_config=cvae_loss_weight_dict,
                optimizer=optimizer,
                average_meter=average_meter,
                device=device)
