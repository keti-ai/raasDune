import logging
from typing import List

from .dataset import ConcatDatasetv2, DatasetGroup
from .dino2 import AVAILABLE_DATASETS as AVAILABLE_DATASETS_DINO2
from .imagenet import AVAILABLE_DATASETS as AVAILABLE_DATASETS_IMAGENET
from .mast3r import AVAILABLE_DATASETS as AVAILABLE_DATASETS_MAST3R
from .multihmr import AVAILABLE_DATASETS as AVAILABLE_DATASETS_MULTIHMR
from .transform import get_test_transform, get_train_transform


logger = logging.getLogger(__name__)

AVAILABLE_DATASETS = {
    **AVAILABLE_DATASETS_IMAGENET,
    **AVAILABLE_DATASETS_MULTIHMR,
    **AVAILABLE_DATASETS_DINO2,
    **AVAILABLE_DATASETS_MAST3R,
}

TEACHER_TO_DATASETS = {
    "multihmr": AVAILABLE_DATASETS_MULTIHMR,
    "mast3r": AVAILABLE_DATASETS_MAST3R,
    "dino2": AVAILABLE_DATASETS_DINO2,
}


def dataset_to_teacher(dataset: str) -> str:
    if dataset in list(AVAILABLE_DATASETS_DINO2.keys()) + ["in1k"]:
        return "dino2"
    elif dataset in AVAILABLE_DATASETS_MAST3R:
        return "mast3r"
    elif dataset in AVAILABLE_DATASETS_MULTIHMR:
        return "multihmr"
    raise ValueError(f"Unknown dataset: {dataset}")


def get_dataset(
    dataset_name, split="train", image_size=224, rrc_scale=(0.08, 1.0), color_aug=True
):
    """
    dataset_name can be a list of datasets separated by "," eg in1k,bedlam
    it can also be "teacher_balanced"
    """

    if dataset_name == "teacher_balanced":
        assert split == "train", "teacher_balanced is only available for training"
        return get_teacher_balanced_dataset(
            image_size=image_size, rrc_scale=rrc_scale, color_aug=color_aug
        )

    elif dataset_name == "all":
        all_teacher_datasets = get_all_teacher_datasets(split)
        dataset_name = ",".join(all_teacher_datasets)
        logger.info(
            "Using all datasets ('{}') for '{}' split".format(dataset_name, split)
        )

    elif dataset_name.startswith("all_except_"):
        excluded_dataset = dataset_name.split("all_except_")[1]
        assert excluded_dataset in AVAILABLE_DATASETS, (
            "Unknown dataset to exclude: " + excluded_dataset
        )
        datasets_to_use = [
            d for d in get_all_teacher_datasets(split) if d != excluded_dataset
        ]
        dataset_name = ",".join(datasets_to_use)
        logger.info(
            "Using all datasets except '{}' ('{}') for '{}' split".format(
                excluded_dataset, dataset_name, split
            )
        )

    elif dataset_name in ["mast3r", "multihmr", "dino2"]:
        _teacher_datasets = TEACHER_TO_DATASETS[dataset_name]
        dataset_name = ",".join(
            d for d in list(_teacher_datasets.keys()) if split in AVAILABLE_DATASETS[d]
        )

    datasets = [
        get_one_dataset(
            dname,
            split,
            image_size=image_size,
            rrc_scale=rrc_scale,
            color_aug=color_aug,
        )
        for dname in dataset_name.split(",")
    ]

    if split == "train":
        # at training, we return a ConcatDataset except if there is a single dataset
        if len(datasets) == 1:
            return datasets[0]
        else:
            dataset = ConcatDatasetv2(datasets)
            return dataset
    else:
        # at validation/test, we return a list of datasets in any case
        return [d for d in datasets if len(d) > 0]


def get_one_dataset(
    dataset_name, split, image_size=224, rrc_scale=(0.08, 1.0), color_aug=True
):

    assert dataset_name in AVAILABLE_DATASETS, "Unknown dataset_name: " + dataset_name
    expected_length = AVAILABLE_DATASETS.get(dataset_name).get(split)

    transform = (
        get_train_transform(
            image_size=image_size, rrc_scale=rrc_scale, color_aug=color_aug
        )
        if split == "train"
        else get_test_transform(image_size)
    )

    dataset_getter = AVAILABLE_DATASETS.get(dataset_name).get("getter")
    logger.info(
        "Loading dataset {} (image_size:{}, rrc_scale:{})".format(
            dataset_name, image_size, rrc_scale
        )
    )
    dataset = dataset_getter(dataset_name, split, transform=transform)

    if len(dataset) != expected_length:
        raise Exception(
            "Unexpected length of {} for split {} of dataset {}, instead of {}".format(
                len(dataset), split, dataset.dataset_name, expected_length
            )
        )

    return dataset


def get_teacher_balanced_dataset(image_size=224, rrc_scale=(0.08, 1.0), color_aug=True):
    # Concatenate all datasets for each teacher

    logger.info("=> Loading DINOv2 datasets")
    dino2_datasets = ConcatDatasetv2(
        [
            get_one_dataset(
                k,
                "train",
                image_size=image_size,
                rrc_scale=rrc_scale,
                color_aug=color_aug,
            )
            for k in AVAILABLE_DATASETS_DINO2.keys()
        ]
    )

    logger.info("=> Loading MAST3R datasets")
    mast3r_datasets = ConcatDatasetv2(
        [
            get_one_dataset(
                k,
                "train",
                image_size=image_size,
                rrc_scale=rrc_scale,
                color_aug=color_aug,
            )
            for k in AVAILABLE_DATASETS_MAST3R.keys()
        ]
    )

    logger.info("=> Loading Multi-HMR datasets")
    multihmr_datasets = ConcatDatasetv2(
        [
            get_one_dataset(
                k,
                "train",
                image_size=image_size,
                rrc_scale=rrc_scale,
                color_aug=color_aug,
            )
            for k in AVAILABLE_DATASETS_MULTIHMR.keys()
        ]
    )

    # Create a dataset group for all teachers
    dataset_dict = {
        "dino2": dino2_datasets,
        "mast3r": mast3r_datasets,
        "multihmr": multihmr_datasets,
    }

    dataset = DatasetGroup(dataset_dict)

    return dataset


def get_all_teacher_datasets(split: str = "train") -> List[str]:
    assert split in ["train", "val"], "split must be 'train' or 'val', got: {split}"
    dsets = list(AVAILABLE_DATASETS_DINO2.keys())
    dsets.extend(list(AVAILABLE_DATASETS_MAST3R.keys()))
    dsets.extend(list(AVAILABLE_DATASETS_MULTIHMR.keys()))
    dsets = [d for d in dsets if split in AVAILABLE_DATASETS[d]]
    return dsets
