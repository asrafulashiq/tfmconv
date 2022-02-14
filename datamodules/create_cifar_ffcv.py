from typing import List

import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

if __name__ == '__main__':
    datasets = {
        'train':
        torchvision.datasets.CIFAR10('data/cifar10', train=True,
                                     download=True),
        'test':
        torchvision.datasets.CIFAR10('data/cifar10',
                                     train=False,
                                     download=True)
    }
    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'data/cifar_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)
