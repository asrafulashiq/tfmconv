import os
import torchvision
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter

if __name__ == '__main__':
    data_dir = 'data/imagenet1k'
    datasets = {
        'train':
        torchvision.datasets.ImageFolder(os.path.join(
            data_dir, "train")),
        'test':
        torchvision.datasets.ImageFolder(os.path.join(
            data_dir, "val")),
    }
    for (name, ds) in datasets.items():
        os.makedirs('data/imagenet1k_ffcv', exist_ok=True)
        writer = DatasetWriter(f'data/imagenet1k_ffcv/imagenet1k_{name}.beton', {
            'image': RGBImageField(max_resolution=256),
            'label': IntField()
        }, num_workers=32)
        writer.from_indexed_dataset(ds)
