from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.loggers import LightningLoggerBase
from loguru import logger
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
import os
from tqdm import tqdm
from colorama import init, Fore


class CustomLogger(LightningLoggerBase):
    def __init__(self,
                 name,
                 save_dir,
                 test=False,
                 disable_logfile=False,
                 version=None,
                 *args,
                 **kwargs):
        super().__init__()
        self._logger = logger
        self._name = name
        self._save_dir = save_dir
        self._version = version
        if test:
            self._save_dir = "test_" + save_dir
        self._experiment = None
        self.disable_logfile = disable_logfile

        self.csv_logger = None

    def _create_logger(self):
        # CLI logger
        self._logger.remove()
        self._logger.configure(handlers=[
            dict(
                sink=lambda msg: tqdm.write(msg, end=''),
                level='DEBUG',
                colorize=True,
                format=
                "<green>{time: MM-DD at HH:mm}</green> <level>{message}</level>",
                enqueue=True),
        ])

        # add file handler for training mode
        if not self.disable_logfile:
            os.makedirs(self.log_dir, exist_ok=True)
            logfile = os.path.join(self.log_dir, "log.txt")
            self._logger.warning(f"Log to file {logfile}")
            self._logger.add(sink=logfile,
                             mode='w',
                             format="{time: MM-DD at HH:mm} | {message}",
                             level="DEBUG",
                             enqueue=True)

            # os.environ["LOG_DIR"] = str(self.log_dir)

    @property
    def log_dir(self):
        version = self.version if isinstance(
            self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment:
            return self._experiment
        self._create_logger()
        self._experiment = self._logger
        return self._experiment

    @staticmethod
    def _handle_value(value):
        if isinstance(value, torch.Tensor):
            try:
                return value.item()
            except ValueError:
                return value.mean().item()
        return value

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if len(metrics) == 0:
            return

        metrics_str = "  ".join([
            f"{k}: {self._handle_value(v):<4.6f}" for k, v in metrics.items()
            if k != 'epoch'
        ])

        if metrics_str.strip() == '':
            return

        if step is not None:
            metrics_str = f"step: {step:<6d} :: " + metrics_str
        if 'epoch' in metrics:
            metrics_str = f"epoch: {int(metrics['epoch']):<4d}  " + metrics_str
        self.experiment.info(metrics_str)

    @rank_zero_only
    def info_metrics(self, metrics, epoch=None, step=None, level='INFO'):
        if isinstance(metrics, str):
            self.experiment.info(metrics)
            return

        _str = ""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            _str += f"{k}= {v:<4.4f}  "
        self.experiment.log(level,
                            f"epoch {epoch: <4d}: step {step:<6d}:: {_str}")

    @rank_zero_only
    def log(self, msg, level='DEBUG'):
        self.experiment.log(level, msg)

    @property
    def name(self):
        return self._name

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self._save_dir, self.name)

        if not os.path.isdir(root_dir):
            logger.warning('Missing logger folder: %s', root_dir)
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir,
                                          d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    def dict_to_str(self, d, parent=""):
        import collections
        if not isinstance(d, collections.abc.MutableMapping):
            return str(d)
        _str = ""
        for k in sorted(d):
            v = d[k]
            if parent:
                k = parent + "." + k
            v = self.dict_to_str(v, parent=k)

            _str += Fore.LIGHTCYAN_EX + str(k) + "="
            _str += Fore.WHITE + str(v) + ", "
        return _str

    @rank_zero_only
    def log_hyperparams(self, params):
        # _str = self.dict_to_str(dict(params))
        from omegaconf import OmegaConf
        try:
            self.experiment.info(OmegaConf.to_yaml(params, resolve=True))
        except (AssertionError, ValueError):
            self.experiment.info(params)
        return

    @property
    def root_dir(self) -> str:
        if not self.name:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)

    @property
    def save_dir(self):
        return self._save_dir
