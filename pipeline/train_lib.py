import os
from absl import logging
from typing import Callable, Dict, Optional, Sequence, Tuple
import gin.torch
import torch
from tqdm import tqdm
from .configure_param import get_cvae_param, get_dataset_param
from datasets.dataset_lib import Dataset
from datasets.data_io_lib import MASK_KEY
from datasets.data_io_lib import IMAGE_KEY, GROUND_TRUTH_KEY, MASK_KEY
from models.model_lib import ConditionalVAE, DISCRETE_ENCODER, GAUSSIAN_ENCODER
from utils.plotting_lib import AverageMeter, log_scalar_dict
from utils.loss_lib import combine_fidelity_losses, combine_loss, get_current_loss_config
from .eval_lib import eval


def train_epoch(model: ConditionalVAE, data_loader: torch.utils.data.DataLoader,
                fidelity_loss_fn: Callable[
                    [torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
                    torch.Tensor], loss_weight_config: Dict[str, float],
                optimizer: torch.optim.Optimizer, average_meter: AverageMeter,
                device: str):#, pbar: tqdm, epoch_loss = 0, sample_cnt = 0):

  Tensor = torch.cuda.FloatTensor if device == torch.device("cuda") else torch.FloatTensor

  def train_step(batch: Dict[str, torch.Tensor]):#, epoch_loss, sample_cnt):
    inputs = batch[IMAGE_KEY].to(device).type(Tensor)
    ground_truth = batch[GROUND_TRUTH_KEY].to(device).type(Tensor)
    mask = None
    if MASK_KEY in batch:
      mask = batch[MASK_KEY].to(device).type(Tensor)

    prediction = model.forward(inputs=inputs, label=ground_truth)

    if model.get_encoder_class() == DISCRETE_ENCODER:
      code_indices = model.posterior_distribution[1]
      model._posterior_encoder._code_book.record_code_usage_for_batch(code_indices)

    loss_dict = {}
    loss_dict["kl"] = model.compute_kl_divergence()
    loss_dict["data_fidelity"] = fidelity_loss_fn(prediction, ground_truth,
                                                  mask)
    loss_dict["regularization"] = model.compute_regularization_loss()

    average_meter.update(loss_dict)

    loss = combine_loss(loss_dict, loss_weight_config)

    # epoch_loss += loss.item()
    # sample_cnt += 1
    # pbar.set_postfix(**{'loss(batch)': loss.item(), 'epoch avg loss:': epoch_loss / sample_cnt})

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # pbar.update(data_loader.batch_size) 

  average_meter.reset()
  if model.get_encoder_class() == DISCRETE_ENCODER:
    model._posterior_encoder._code_book.reset_usage_summary()

  for step_index, batch in enumerate(data_loader):
    train_step(batch)#, epoch_loss, sample_cnt)
    if step_index % 100 == 0:
      log_scalar_dict(step_index, average_meter.get_moving_average_dict())

  if model.get_encoder_class() == DISCRETE_ENCODER:
    model._posterior_encoder._code_book.log_code_usage()


def adaptive_code_book_initialization(model: ConditionalVAE,
                                      data_loader: torch.utils.data.DataLoader,
                                      device: torch.device):
  Tensor = torch.cuda.FloatTensor if device == torch.device("cuda") else torch.FloatTensor
  code_stat = AverageMeter()
  
  for _, batch in enumerate(data_loader):
    inputs = batch[IMAGE_KEY].to(device).type(Tensor)
    ground_truth = batch[GROUND_TRUTH_KEY].to(device).type(Tensor)

    _ = model.forward(inputs=inputs, label=ground_truth)
    code = model.posterior_sample(use_random=False)

    code_stat.update({
        "variance": code.pow(2).mean(dim=0, keepdim=True).cpu().detach().numpy(),
        "mean": code.mean(dim=0, keepdim=True).cpu().detach().numpy()
    })

  initial_std = torch.from_numpy(code_stat.get_average_dict()["variance"]).pow(0.5).squeeze(3).squeeze(2)
  initial_mean = torch.from_numpy(code_stat.get_average_dict()["mean"]).squeeze(3).squeeze(2)

  model.preprocess(mean=initial_mean.transpose(0,1), std=initial_std.transpose(0,1))



@gin.configurable
def train(batch_size: int,
          num_epochs: int,
          base_save_path: str,
          fidelity_loss_config_dict: Dict[str, float],
          loss_weight_config_list: Sequence[Tuple[int, Dict[str, float]]],
          initial_learning_rate: float,
          learning_rate_milestones: Dict[int, float],
          eval_epoch_interval: int = 1,
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
  dataset_param = get_dataset_param()
  dataset_name = dataset_param["dataset_name"]
  dataset = Dataset(**dataset_param)

  model_param = get_cvae_param()
  model_name = model_param["encoder_class"]

  if model_name == GAUSSIAN_ENCODER:
    eval_use_random = True
  elif model_name == DISCRETE_ENCODER:
    eval_use_random = False

  model = ConditionalVAE(**model_param)
  model.preprocess(**{'mean': 0, 'std': 1})
  # initialize codebook firstly
  model.to(device)
  train_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True,
                                             sampler=None)
  optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
  fidelity_loss_fn = combine_fidelity_losses(fidelity_loss_config_dict)
  average_meter = AverageMeter()

  dataset_model_token = "%s_%s" % (dataset_name, model_name)

  # perform adaptive code book initialization for discrete posterior encoder
  if model.get_encoder_class() == DISCRETE_ENCODER:
    adaptive_code_book_initialization(model, train_loader, device)
  # transport adaptive code book from CPU to CUDA
  model.to(device)

  milestone_index = 1
  for epoch_index in range(num_epochs):
    loss_weight_dict = get_current_loss_config(epoch_index, loss_weight_config_list)
    if epoch_index + 1 in learning_rate_milestones:
      logging.info("Using learning rate %f" %
                   (learning_rate_milestones[milestone_index]))
      for pg in optimizer.param_groups:
        pg["lr"] = learning_rate_milestones[milestone_index]
    model.train()
    milestone_index += 1
    # with tqdm(total=num_epochs, desc='Epoch {}/{}'.format(epoch_index, num_epochs), unit='img') as pbar:
    train_epoch(model=model,
                data_loader=train_loader,
                fidelity_loss_fn=fidelity_loss_fn,
                loss_weight_config=loss_weight_dict,
                optimizer=optimizer,
                average_meter=average_meter,
                device=device)#, pbar=pbar)
    logging.info("epoch %d" %(epoch_index))
    if not os.path.isdir(os.path.join(base_save_path, "train", dataset_model_token)):
      os.makedirs(os.path.join(base_save_path, "train", dataset_model_token))

    if (epoch_index + 1) % eval_epoch_interval == 0 or (epoch_index +
                                                        1) == num_epochs:

      torch.save(
          model.state_dict(),
          os.path.join(base_save_path, "train", dataset_model_token,
                      "epoch_%d.pth" %(epoch_index)))
      logging.info("saving epoch%d for %s model, %s" %
              (epoch_index, model_name, dataset_name))

      eval(model,
          check_point_path=None,
          use_random=eval_use_random,
          top_k=eval_top_k_samples,
          num_cvae_sample=eval_num_samples,
          base_save_path=base_save_path,
          epoch_index=epoch_index)

