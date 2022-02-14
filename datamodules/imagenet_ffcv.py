from typing import Any, List, Optional
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
import torchvision
from timm.models.layers import to_2tuple


from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToTorchImage, RandomHorizontalFlip
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms.common import Squeeze



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
        crop_pct: Optional[float] = None,
        strategy: str = "ddp",
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
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.crop_pct = crop_pct

        self.distributed = True if strategy == "ddp" else False

        self.train_beton = os.path.join(self.data_dir, "imagenet1k_train.beton")
        self.val_beton = os.path.join(self.data_dir, "imagenet1k_test.beton")

    @property
    def num_classes(self) -> int:
        return 1000

    def prepare_data(self) -> None:
        assert os.path.exists(self.train_beton), "Train beton file not found"
        assert os.path.exists(self.val_beton), "Val beton file not found"

    def train_dataloader(self) -> DataLoader:
        """Uses the train split of imagenet2012"""
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]
        image_pipeline: self.train_transform(
        ) if self.train_transforms is None else self.train_transforms
        
        loader = Loader(
            self.train_beton,
            batch_size=self.batch_size,
            order=OrderOption.RANDOM,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            distributed=self.distributed,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            }
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]
        image_pipeline: self.train_transform(
        ) if self.train_transforms is None else self.train_transforms

        loader = Loader(
            self.val_beton,
            batch_size=self.batch_size,
            order=OrderOption.SEQUENTIAL,
            num_workers=self.num_workers,
            drop_last=False,
            distributed=self.distributed,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            }
        )
        return loader


    def test_dataloader(self) -> DataLoader:
        """Uses the validation split of imagenet2012 for testing."""
        transforms = self.val_transform(
        ) if self.test_transforms is None else self.test_transforms

        dataset = torchvision.datasets.ImageFolder(os.path.join(
            self.data_dir, "val"),
                                                   transform=transfo