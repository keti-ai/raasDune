# For logging problematic images
PROBLEMATIC_IMAGES_LOG_FILE = "/path/to/some/json/file.json"

# List of directories where the ImageNet-1K dataset can be found.
# We monitor distillation loss on this dataset for debugging
# it is not necessary for training DUNE models.
IN1K_DIRS = [
    "/path/to/ilsvrc2012",
]

# Path to a pickle file which contains a list of ImageNet-19K images.
# Traversing the entire dataset is slow, so we use a precomputed list.
# This file is not included in the repository, but can be generated
# by first loading up the dataset using torchvision then saving the list of images to a pickle file.
IN19K_PKL_PATH = "/path/to/imagenet19k/images.pkl"
DINOV2_DATASET_PATHS = {
    "in19k": IN19K_PKL_PATH,
    "gldv2": "/path/to/google-landmarks-dataset-v2",
    "mapillarystreet": "/path/to/mapillary-street",
}

# To prepare the Mast3r datasets, see here:
# https://github.com/naver/mast3r?tab=readme-ov-file#datasets
MAST3R_DATASET_PATHS = {
    "ARKitScenesV2": "",
    "BlendedMVS": "",
    "Co3d_v3": "",
    "DL3DV": "",
    "Habitat512": "",
    "MegaDepthDense": "",
    "NLK_MVS": "",
    "Niantic": "",
    "ScanNetppV2": "",
    "TartanAir": "",
    "Unreal4K": "",
    "VirtualKitti": "",
    "WildRgb": "",
}
MAST3R_CACHE_DIR = "/path/to/mast3r/cache/dir"

HMR_DATASET_PATHS = {
    "bedlam": "/path/to/BEDLAM",
    "agora": "/path/to/agora/images",
    "cuffs": "/path/to/CUFFS",
    "ubody": "/path/to/UBody",
    "ubody_pkl": "/path/to/ubody.pkl",
}
