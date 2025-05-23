import logging
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Union

from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import ImageFolder

from .paths import PROBLEMATIC_IMAGES_LOG_FILE
from .utils import add_str_to_jsonfile


logger = logging.getLogger()


def my_pil_loader(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


class ImageFolderV2(ImageFolder):
    """
    Same as ImageFolder, but also returns a dataset_name as third output
    """

    def __init__(self, dataset_name, *args, **kwargs):
        if "loader" not in kwargs:
            kwargs["loader"] = my_pil_loader

        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name

    def __repr__(self):
        lines = super().__repr__().split("\n")
        lines.insert(1, " " * self._repr_indent + f"Dataset Name: {self.dataset_name}")
        return "\n".join(lines)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        try:
            # Modify the __getitem__ for DatasetFolder
            # instead of calling it directly by "sample, target = super().__getitem__(index)"
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        except Exception as e:
            logger.info(f"Error in {self.dataset_name} at index {index}: {e}")
            add_str_to_jsonfile(PROBLEMATIC_IMAGES_LOG_FILE, path)
            return None

        return sample, target, self.dataset_name


class ImageOneFolderV2(ImageFolderV2):
    """
    ImageFolder is nice but assume one folder per class. This class does the same but where all images are directly in the root folder.
    """

    def find_classes(
        self, directory: Union[str, Path]
    ) -> Tuple[List[str], Dict[str, int]]:
        return [""], {"": -1}


class ImageListV2(Dataset):

    def __init__(
        self, dataset_name, root, imlist, transform=None, loader=my_pil_loader
    ):
        self.dataset_name = dataset_name
        self.root = root
        self.imlist = imlist
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, index):
        impath = os.path.join(self.root, self.imlist[index])

        sample = self.loader(impath)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, -1, self.dataset_name

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Dataset Name: {self.dataset_name}"]
        body += [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)


class ConcatDatasetv2(ConcatDataset):
    def __repr__(self):
        head = ["Dataset " + self.__class__.__name__]
        body = [f"Number of data points: {self.__len__()} ({self.cumulative_sizes})"]
        body2 = [repr(d).replace("\n", "\n\t") for d in self.datasets]
        return "\n\t".join(head + body + body2)


class DatasetGroup(Dataset):
    """
    Given a group of datasets (in a form of dictionary),
    returns one sample from each dataset.
    Indexing of the group is determined by the largest dataset in the group.
    The index of the smaller ones are cyclic.
    Warning: "Residual indices" for the smaller datasets are randomly shuffled.
    """

    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__()
        self.datasets = datasets
        self.res_inds = {k: deque([]) for k in datasets.keys()}

    def init_group_res_index(self, dset_key):
        assert dset_key in self.res_inds

        if len(self.res_inds[dset_key]) > 0:
            return

        order = list(range(len(self.datasets[dset_key])))
        random.shuffle(order)
        self.res_inds[dset_key] = deque(order)

    def __repr__(self):
        head = [self.__class__.__name__]
        body = [f"Number of data points: {self.__len__()}"]
        body2 = [
            "{}: {}".format(k, repr(v).replace("\n", "\n\t"))
            for k, v in self.datasets.items()
        ]
        return "\n\t".join(head + body + body2)

    def __len__(self):
        # Indexing is determined by the largest group
        return max(len(dataset) for dataset in self.datasets.values())

    def __getitem__(self, idx):
        samples = []
        for dset_key, dataset in self.datasets.items():
            i = idx % len(dataset)

            # residual index for the smaller groups
            if idx >= (len(self) // len(dataset)) * len(dataset):
                self.init_group_res_index(dset_key)
                i = self.res_inds[dset_key].pop()

            samples.append(dataset[i])

        return samples


class EmptyDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __repr__(self):
        return "{}(dataset_name={}, len={})".format(
            self.__class__.__name__, self.dataset_name, len(self)
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset has no data")
