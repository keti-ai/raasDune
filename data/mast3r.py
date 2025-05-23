import os
import pickle

from .dataset import ImageListV2
from .paths import MAST3R_DATASET_PATHS, MAST3R_CACHE_DIR


def get_mast3r_dataset(
    dataset_name,
    split,
    transform,
):
    return Mast3rDataset(dataset_name, split=split, transform=transform)


AVAILABLE_DATASETS = {
    "ARKitScenesV2": {"train": 456_108, "val": 5_307, "getter": get_mast3r_dataset},
    "BlendedMVS": {"train": 98_937, "val": 5_094, "getter": get_mast3r_dataset},
    "Co3d_v3": {"train": 185_100, "val": 5_000, "getter": get_mast3r_dataset},
    "DL3DV": {"train": 208_800, "val": 5_000, "getter": get_mast3r_dataset},
    "Habitat512": {"train": 284_965, "val": 5_035, "getter": get_mast3r_dataset},
    "MegaDepthDense": {"train": 36_949, "val": 3_682, "getter": get_mast3r_dataset},
    "Niantic": {"train": 41_300, "val": 4_600, "getter": get_mast3r_dataset},
    "ScanNetppV2": {"train": 60_188, "val": 5_031, "getter": get_mast3r_dataset},
    "TartanAir": {"train": 136_225, "val": 10_000, "getter": get_mast3r_dataset},
    "Unreal4K": {"train": 14_386, "val": 1_988, "getter": get_mast3r_dataset},
    "VirtualKitti": {"train": 1_200, "val": 300, "getter": get_mast3r_dataset},
    "WildRgb": {"train": 224_400, "val": 5_000, "getter": get_mast3r_dataset},
}


class Mast3rDataset(ImageListV2):

    def __init__(self, dataset_name, split="train", transform=None):
        super().__init__(dataset_name, "", [], transform=transform)

        with open(
            os.path.join(MAST3R_CACHE_DIR, dataset_name + "_" + split + "_impaths.pkl"),
            "rb",
        ) as fid:
            self.root, self.imlist = pickle.load(fid)

        if not os.path.isdir(self.root):
            self.root = MAST3R_DATASET_PATHS[dataset_name]
            assert os.path.isdir(self.root), "{} root not found ({})".format(
                dataset_name, self.root
            )
