import os

from .dataset import ImageFolderV2, ImageOneFolderV2
from .imagenet import get_imagenet
from .paths import DINOV2_DATASET_PATHS


##################################################
# Dataset getters


def get_gldv2(dataset_name, split, transform, **kwargs):
    assert split in ["train", "val"], "split must be 'train' or 'val', got: {split}"

    target_transform = lambda t: -1

    dataset_path = DINOV2_DATASET_PATHS[dataset_name]
    data_dir = os.path.join(dataset_path, "train" if split == "train" else "test")
    assert os.path.isdir(data_dir), data_dir

    dataset = ImageFolderV2(
        dataset_name,
        data_dir,
        transform=transform,
        target_transform=target_transform,
    )

    return dataset


def get_mapillarystreet(dataset_name, split, transform, **kwargs):
    assert split in ["train", "val"], "split must be 'train' or 'val', got: {split}"

    dataset_path = DINOV2_DATASET_PATHS[dataset_name]
    data_dir = os.path.join(dataset_path, "train_val" if split == "train" else "test")
    assert os.path.isdir(data_dir), data_dir

    dataset = ImageOneFolderV2(
        dataset_name,
        data_dir,
        transform=transform,
    )

    return dataset


AVAILABLE_DATASETS = {
    "in19k": {
        "train": 13_153_480,
        "val": 0,
        "getter": get_imagenet,
    },  # 20 problematic images removed
    "gldv2": {
        "train": 4_132_914,
        "val": 117_577,
        "getter": get_gldv2,
    },
    "mapillarystreet": {
        "train": 1_205_907,
        "val": 23_943,
        "getter": get_mapillarystreet,
    },
}
##################################################
