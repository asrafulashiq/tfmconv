# type: ignore[override]
from typing import Any, Callable
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
import torchvision
import torchvision.transforms as transform_lib


class ImagenetDataModule(LightningDataModule):
    name = "imagenet"

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        num_workers: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 1000

    def train_dataloader(self) -> DataLoader:
        """Uses the train split of imagenet2012"""
        transforms = self.train_transform(
        ) if self.train_transforms is None else self.train_transforms

        dataset = torchvision.datasets.ImageFolder(os.path.join(
            self.data_dir, "train"),
                                                   transform=transforms)
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        transforms = self.val_transform(
        ) if self.val_transforms is None else self.val_transforms

        dataset = torchvision.datasets.ImageFolder(os.path.join(
            self.data_dir, "val"),
                                                   transform=transforms)
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """Uses the validation split of imagenet2012 for testing."""
        transforms = self.val_transform(
        ) if self.test_transforms is None else self.test_transforms

        dataset = torchvision.datasets.ImageFolder(os.path.join(
            self.data_dir, "val"),
                                                   transform=transforms)
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def train_transform(self) -> Callable:
        """The standard imagenet transforms.
        """
        preprocessing = transform_lib.Compose([
            transform_lib.RandomResizedCrop(self.image_size),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
            imagenet_normalization(),
        ])

        return preprocessing

    def val_transform(self) -> Callable:
        """The standard imagenet transforms for validation.
        """

        preprocessing = transform_lib.Compose([
            transform_lib.Resize(self.image_size + 32),
            transform_lib.CenterCrop(self.image_size),
            transform_lib.ToTensor(),
            imagenet_normalization(),
        ])
        return preprocessing