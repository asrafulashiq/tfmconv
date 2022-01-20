from typing import MutableSequence, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import os
import shutil
import math
from pathlib import Path

import torch


class CheckpointCallback(pl.callbacks.ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        try:
            self.save_checkpoint(trainer)
        except pl.utilities.exceptions.MisconfigurationException:
            return

    @rank_zero_only
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.every_n_epochs == 0 and pl_module.hparams.test is False:
            last_filepath = os.path.join(self.dirpath, "last.ckpt")
            trainer.save_checkpoint(last_filepath, self.save_weights_only)

        if (self.every_n_epochs is not None
                and trainer.current_epoch % self.every_n_epochs == 0):
            _filepath = os.path.join(self.dirpath,
                                     f"epoch_{trainer.current_epoch}.ckpt")
            trainer.save_checkpoint(_filepath, self.save_weights_only)

    @rank_zero_only
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # save best checkpoint as 'best.ckpt'
        if hasattr(self, 'best_model_path') and os.path.exists(
                self.best_model_path):
            best_path = os.path.join(self.dirpath, "best.ckpt")
            shutil.copy(self.best_model_path, best_path, follow_symlinks=True)

        if hasattr(pl_module.hparams, 'feature_keys'):
            try:
                fkeys = pl_module.hparams.feature_keys
                savekeys = pl_module.hparams.save_keys
                assert isinstance(fkeys, MutableSequence)

                for fk, sk in zip(fkeys, savekeys):
                    _dict = get_state_dict_for_key(fk, pl_module.state_dict())
                    _filepath = os.path.join(self.dirpath, f"{sk}.ckpt")
                    torch.save(_dict, _filepath)
            except Exception:
                print(f"Exception while saving keys")


def get_state_dict_for_key(key, state_dict):
    feature_name = key
    newmodel = {}
    for k, v in state_dict.items():
        if not k.startswith(f"{feature_name}."):
            continue
        old_k = k
        k = k.replace(f"{feature_name}.", "")

        print(old_k, "->", k)
        newmodel[k] = v
    res = {"state_dict": newmodel, "__author__": "ash"}
    return res


class LRScheduler(pl.Callback):

    def __init__(self,
                 initial_lr=0.03,
                 use_cosine_scheduler=False,
                 schedule=None,
                 max_epochs=50):
        super().__init__()
        self.lr = initial_lr
        self.use_cosine_scheduler = use_cosine_scheduler
        if schedule:
            self.schedule = schedule
        else:
            self.schedule = (int(max_epochs * 1 / 3), int(max_epochs * 2 / 3))
        self.max_epochs = max_epochs

    def on_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        lr = self.lr

        if self.use_cosine_scheduler:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / self.max_epochs))
        else:  # stepwise lr schedule
            for milestone in self.schedule:
                lr *= 0.1 if epoch >= milestone else 1.

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr