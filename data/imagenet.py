import logging
import os
import pickle

from torchvision.datasets import VisionDataset

from .dataset import EmptyDataset, ImageFolderV2, my_pil_loader
from .paths import IN1K_DIRS, IN19K_PKL_PATH, PROBLEMATIC_IMAGES_LOG_FILE
from .utils import get_first_available_dir, add_str_to_jsonfile


logger = logging.getLogger()


##################################################
# Dataset getters


def get_imagenet(dataset_name, split, transform, **kwargs):
    assert (
        dataset_name in AVAILABLE_DATASETS
    ), "Unknown ImageNet dataset '{}', expected: '{}'".format(
        dataset_name, list(AVAILABLE_DATASETS.keys())
    )
    assert split in ["train", "val"], "split must be 'train' or 'val', got: {split}"

    if dataset_name.startswith("in19k") and split == "val":
        logging.info(
            "No validation set for {}, returning empty dataset".format(dataset_name)
        )
        return EmptyDataset(dataset_name)

    if dataset_name == "in1k":
        dataset_class = ImageFolderV2
        data_path = get_first_available_dir(IN1K_DIRS)
        data_path = os.path.join(data_path, split)

    elif dataset_name == "in19k":
        dataset_class = ImageNetSubset
        data_path = IN19K_PKL_PATH

    else:
        raise ValueError("Unknown ImageNet dataset: {}".format(dataset_name))

    assert os.path.exists(data_path), data_path
    dataset = dataset_class(dataset_name, data_path, transform=transform)

    return dataset


AVAILABLE_DATASETS = {
    "in1k": {"train": 1_281_167, "val": 50_000, "getter": get_imagenet},
    "in19k": {
        "train": 13_153_480,
        "val": 0,
        "getter": get_imagenet,
    },  # 20 problematic images removed
}

##################################################


class ImageNetSubset(VisionDataset):
    def __init__(self, dataset_name, subset_path, transform=None, imroot=""):
        self.dataset_name = dataset_name
        self.samples = pickle.load(open(subset_path, "rb"))
        self.loader = my_pil_loader
        self.transform = transform
        self.imroot = imroot

        # for compatibility with VisionDataset.__repr__
        self.root = imroot
        self.transforms = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image_path = self.imroot + image_path

        try:
            image = self.loader(image_path)
        except Exception as e:
            logger.error("ERROR while loading image {}".format(image_path))
            logger.error("{}".format(e))
            add_str_to_jsonfile(PROBLEMATIC_IMAGES_LOG_FILE, image_path)
            return None

        image = self.transform(image) if self.transform else image

        return image, target, self.dataset_name
