import os
from typing import Any, Callable, List, Optional, Sequence, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
import torch
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Convert, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze


class CIFAR10DataModule(VisionDataModule):
    """
    Specs:
        - 10 classes (1 per class)
        - Each image is (3 x 32 x 32)

    Standard CIFAR10, train, val, test splits and transforms
    """
    name = "cifar10"
    dataset_cls = CIFAR10
    dims = (3, 32, 32)

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

        self.train_beton_path = os.path.join(self.data_dir, "cifar10_train.beton")
        self.val_beton_path = os.path.join(self.data_dir, "cifar10_test.beton")

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        assert os.path.exists(self.train_beton_path)
        assert os.path.exists(self.val_beton_path)

    def setup(self, stage):
        pass

    def train_dataloader(self, *args: Any, **kwargs: Any):
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]
        image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),        
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float16),
            transform_lib.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)]
        
        loader = Loader(
            self.train_beton_path,
            batch_size=self.batch_size,
            order=OrderOption.RANDOM if self.shuffle else OrderOption.SEQUENTIAL,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            distributed=False,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            }
        )
        return loader

    def val_dataloader(self, *args: Any, **kwargs: Any):
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]
        image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),        
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float16),
            transform_lib.Normalize(self.CIFAR_MEAN, self.CIFAR_STD)]
        
        loader = Loader(
            self.val_beton_path,
            batch_size=self.batch_size,
            order=OrderOption.SEQUE