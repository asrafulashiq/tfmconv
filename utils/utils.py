from typing import List
import numpy as np
from omegaconf import OmegaConf
import torch
import os
import re
import subprocess

import rich.syntax
import rich.tree
from pytorch_lightning.utilities import rank_zero_only
from typing import List, Sequence
from omegaconf import DictConfig, OmegaConf

# NOTE reguster resolver for step lr scheduler, requires omegaconf: 2.1.0.dev26
OmegaConf.register_new_resolver("multiply",
                                lambda x, y: int(float(x) * float(y)))


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse',
                                        'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return None


def get_git_branch() -> str:
    try:
        return subprocess.check_output(
            'git rev-parse --abbrev-ref HEAD'.split()).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return None


def to_numpy(x: torch.Tensor):
    if not isinstance(x, torch.Tensor):
        return x
    if x.is_cuda:
        x = x.cpu()
    return x.data.numpy()


def set_environment_variables_for_nccl_backend(single_node=False,
                                               master_port=6105):
    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]

        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = "54965"

    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NODE_RANK"] = os.environ["PMI_RANK"]


def set_environment_variables():
    os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
    os.environ["MASTER_PORT"] = "6105"

    # node rank is the world rank from mpi run
    os.environ["NODE_RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    print("MASTER_ADDR = {}".format(os.environ["MASTER_ADDR"]))
    print("MASTER_PORT = {}".format(os.environ["MASTER_PORT"]))
    print("NODE_RANK = {}".format(os.environ["NODE_RANK"]))


def refine_args(params):
    # set backend properly
    if not isinstance(params.trainer.gpus, int):
        if params.trainer.gpus == '-1':
            params.trainer.gpus = -1
            raise ValueError(
                "Use specific number of gpu, DDP not working for gpus=-1")
        else:
            _split = params.trainer.gpus.split(',')
            if len(_split) == 1:
                params.trainer.gpus = int(params.trainer.gpus)
            else:
                list_of_gpu = [int(k) for k in _split if k.isdigit()]
                params.trainer.gpus = list_of_gpu

    if ((isinstance(params.trainer.gpus, list)
         and len(params.trainer.gpus) > 1) or
        (isinstance(params.trainer.gpus, int)
         and params.trainer.gpus > 1)) or (params.trainer.gpus == -1):
        params.trainer.strategy = 'ddp'

    if params.resume and os.path.exists(
            os.path.join(params.trainer.weights_save_path, 'last.ckpt')):
        params.trainer.resume_from_checkpoint = os.path.join(
            params.trainer.weights_save_path, 'last.ckpt')
        if params.test:
            params.ckpt = params.trainer.resume_from_checkpoint

    if 'LOCAL_RANK' in os.environ:
        device_name = f'cuda:{os.environ["LOCAL_RANK"]}'
    else:
        device_name = 'cuda'
    print("GPU Info :", torch.cuda.get_device_properties(device_name))

    return params


def configure_loguru(logger, logfile=None):
    if logfile is not None:
        logger.critical(f"log to {logfile}")
        logger.add(sink=logfile,
                   mode='w',
                   format="{time: MM-DD at HH:mm} | {message}",
                   level="DEBUG",
                   enqueue=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


def get_files(folder: str, skip: int = 1) -> List[str]:
    lists = os.listdir(folder)
    lists = sorted(lists, key=to_key)
    lists = lists[::skip]
    return [os.path.join(folder, path) for path in lists]


def to_key(path):
    """ get int order from file name """
    ints = re.findall(r'\d+', path)
    if len(ints) == 0:
        return -100
    else:
        return int(ints[0])


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = int(np.ceil(warmup_epochs * niter_per_ep))
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(int(np.ceil(epochs * niter_per_ep)) - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule
