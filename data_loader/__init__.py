from .cityscapes import CitySegmentation
from .cityscapes_video import CitySegmentation_video

datasets = {
    'citys': CitySegmentation,
}
datasets_video = {
    'citys': CitySegmentation_video,
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)

def get_segmentation_dataset_video(name, **kwargs):
    """Segmentation Datasets"""
    return datasets_video[name.lower()](**kwargs)