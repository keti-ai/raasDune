import logging
import os
import pickle

from .dataset import (
    EmptyDataset,
    ImageFolderV2,
    ImageListV2,
    ImageOneFolderV2,
    my_pil_loader,
)
from .paths import HMR_DATASET_PATHS


logger = logging.getLogger()


##################################################
# Dataset getters


def get_bedlam(dataset_name, split, transform, **kwargs):
    assert split in ["train", "val"], "split must be 'train' or 'val', got: {split}"

    split_dir = {"train": "training", "val": "validation"}[split]
    dataset_path = HMR_DATASET_PATHS[dataset_name]
    data_dir = os.path.join(dataset_path, split_dir)
    assert os.path.isdir(data_dir), data_dir

    def target_transform(t):
        return -1

    def is_valid_file(x):
        return x.endswith(".png") and not os.path.basename(x).startswith(".")

    def my_pil_loader_with_bedlamfix_rotation(path):
        img = my_pil_loader(path)
        if "closeup" in path:
            img = img.rotate(-90)
        return img

    dataset = ImageFolderV2(
        dataset_name,
        data_dir,
        transform=transform,
        target_transform=target_transform,
        is_valid_file=is_valid_file,
        loader=my_pil_loader_with_bedlamfix_rotation,
    )

    return dataset


def get_agora(dataset_name, split, transform, **kwargs):
    assert split in ["train", "val"], "split must be 'train' or 'val', got: {split}"

    dataset_path = HMR_DATASET_PATHS[dataset_name]
    data_dir = {
        "train": os.path.join(dataset_path, "train"),
        "val": os.path.join(dataset_path, "validation"),
    }[split]

    def is_valid_file(x):
        return not os.path.basename(x).startswith(".")

    dataset = ImageOneFolderV2(
        dataset_name, data_dir, transform=transform, is_valid_file=is_valid_file
    )

    return dataset


def get_cuffs(dataset_name, split, transform, **kwargs):
    assert split in ["train", "val"], "split must be 'train' or 'val', got: {split}"

    if split == "val":
        logging.info(f"Dataset {dataset_name} does not have a {split} split")
        return EmptyDataset(dataset_name)

    dataset_path = HMR_DATASET_PATHS[dataset_name]
    dataset = ImageOneFolderV2(dataset_name, dataset_path, transform=transform)

    return dataset


def get_ubody(dataset_name, split, transform, **kwargs):
    assert split in ["train", "val"], "split must be 'train' or 'val', got: {split}"

    dataset_path = HMR_DATASET_PATHS[dataset_name]
    imroot = os.path.join(dataset_path, "videos")
    pkl_fname = f"{HMR_DATASET_PATHS['ubody_pkl']}/ubody_intra_{'train' if split=='train' else 'test'}.pkl"

    with open(pkl_fname, "rb") as fid:
        annot = pickle.load(fid)

    imlist = sorted(list(annot.keys()))
    dataset = ImageListV2(dataset_name, imroot, imlist, transform=transform)

    return dataset


AVAILABLE_DATASETS = {
    "bedlam": {
        "train": 353_116,
        "val": 31_381,
        "getter": get_bedlam,
    },  # "hidden" training images are not counted
    "agora": {
        "train": 14_314,
        "val": 1_225,
        "getter": get_agora,
    },  # "hidden" training images are not counted
    "cuffs": {"train": 54_944, "val": 0, "getter": get_cuffs},
    "ubody": {"train": 54_234, "val": 2_016, "getter": get_ubody},
}

##################################################
