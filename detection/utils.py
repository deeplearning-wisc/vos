import logging
import numpy as np
import time
import weakref
from typing import Dict
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage,get_event_storage
from detectron2.engine.train_loop import TrainerBase


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:
    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.
    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, data_loader_ood, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.data_loader_ood = data_loader_ood
        self._data_loader_iter_ood = iter(data_loader_ood)
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_ood = next(self._data_loader_iter_ood)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data, data_ood, self.iter)
        losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        device = next(iter(loss_dict.values())).device

        # Use a new stream so these ops don't wait for DDP or backward
        with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict["data_time"] = data_time

            # Gather metrics among all workers for logging
            # This assumes we do DDP-style training, which is currently the only
            # supported method in detectron2.
            all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)
