from typing import List
from utils.utils import refine_args, get_git_revision_hash, get_git_branch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.base import Callback
from utils.helper_slurm import get_argv
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
import argparse


def main(params: DictConfig, *args, **kwargs):
    params = OmegaConf.create(OmegaConf.to_yaml(params, resolve=True))
    params = refine_args(params)
    # if params.print_conf:
    #     print(OmegaConf.to_yaml(params))

    pl.seed_everything(params.seed, workers=True)

    # Init datamodule
    dm = hydra.utils.instantiate(params.dataset.instance)

    # transform
    dm.train_transforms = hydra.utils.instantiate(params.train_transform)
    dm.val_transforms = hydra.utils.instantiate(params.val_transform)
    dm.test_transforms = hydra.utils.instantiate(params.val_transform)

    # Init PyTorch Lightning model ⚡
    lightning_model = hydra.utils.instantiate(params.system,
                                              hparams=params,
                                              _recursive_=False)

    if params.ckpt is not None and params.ckpt != 'none':
        ckpt = torch.load(
            params.ckpt,
            map_location=lambda storage, loc: storage)['state_dict']
        lightning_model.load_state_dict(ckpt)

    logger = hydra.utils.instantiate(params.logger)
    logger.log(
        f"Git || Branch: {get_git_branch()} | hash: {get_git_revision_hash()}")
    logger.log(f"Command: python {get_argv()}")

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = [
        hydra.utils.instantiate(callback_conf)
        for _, callback_conf in params["callbacks"].items()
    ] if "callbacks" in params and params.callbacks else []
    callbacks.append(RichProgressBar())

    trainer = pl.Trainer.from_argparse_args(
        argparse.Namespace(**params.trainer),
        logger=logger,
        callbacks=callbacks,
    )

    if params.test:
        trainer.test(lightning_model, datamodule=dm)
    else:
        return trainer.fit(lightning_model, datamodule=dm)


@hydra.main(config_name="config", config_path="conf")
def hydra_main(cfg: DictConfig):

    is_local_run = (cfg.launcher is None or cfg.launcher.name == "local"
                    or (cfg.launcher.name == "slurm"
                        and cfg.launcher.from_slurm is True))

    if is_local_run:
        main(cfg)
    elif cfg.launcher.name == "slurm":
        from utils.helper_slurm import run_cluster
        run_cluster(cfg, main)
    else:
        raise NotImplementedError(
            f"Launcher {cfg.launcher.name} is not implemented yet!")


if __name__ == "__main__":
    hydra_main()
