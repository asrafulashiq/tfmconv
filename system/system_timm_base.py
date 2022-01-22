from typing import Any
import torch
import torch.nn as nn
import abc
import hydra
from pytorch_lightning import LightningModule
import hydra
import torchmetrics
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

        # module
        self._create_model()

        # metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        # criterion
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)

        # mixup/cutmix
        self.mixup_fn = hydra.utils.instantiate(self.hparams.mixup_fn)

    # ----------------------------------- model ---------------------------------- #
    def _create_model(self):
        """ create encoder and classifier head """
        self.model = hydra.utils.instantiate(self.hparams.model)

    # ----------------------------------- init ----------------------------------- #
    def setup(self, stage):
        if stage == 'fit':
            if self.trainer.world_size > 1:
                num_proc = self.trainer.num_nodes * self.trainer.num_processes
            else:
                num_proc = 1
            self.hparams.optimizer.lr = (
                self.hparams.base_lr * self.hparams.batch_size *
                self.trainer.accumulate_grad_batches *
                num_proc) / self.hparams.base_batch_size
            print(f"Learning rate set to {self.hparams.optimizer.lr}")

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if not hasattr(self, 'lr_schedule'):
            self.setup_scheduler()

    def setup_scheduler(self):
        # ============ init schedulers ... ============
        train_len = self.trainer.num_training_batches
        self.lr_schedule = hydra.utils.instantiate(
            self.hparams.lr_schedule,
            base_value=self.hparams.optimizer.lr,
            niter_per_ep=train_len)

        if self.hparams.wd_schedule:
            self.wd_schedule = hydra.utils.instantiate(
                self.hparams.wd_schedule, niter_per_ep=train_len)
        else:
            self.wd_schedule = None

    def on_train_batch_start(self, batch: Any, batch_idx: int,
                             dataloader_idx: int) -> None:
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)

        # lr scheduling
        optimizer = self.optimizers()
        it = self.global_step
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                if self.wd_schedule is not None:
                    param_group["weight_decay"] = self.wd_schedule[it]

    # --------------------------------- training --------------------------------- #
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mixup_fn:
            x, y = self.mixup_fn(x, y)

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train/loss', loss, prog_bar=True)

        if self.mixup_fn is None:
            acc = self.train_acc(y_hat, y)
            self.log('train/acc', acc, prog_bar=True)

        self.log('lr', self.get_lr(), prog_bar=True)
        return loss

    # ----------------------------------- eval ----------------------------------- #
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.val_acc(y_hat, y)
        self.log('val/loss', loss)
        self.log('val/acc', acc)
        return loss

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def get_lr(self):
        optimizer = self.optimizers()
        return optimizer.param_groups[0]['lr']

    # --------------------------------- optimizer -------------------------------- #
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            filter(lambda p: p.requires_grad, self.parameters()))

        return optimizer
