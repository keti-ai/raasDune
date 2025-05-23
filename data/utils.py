import os
import pickle
import json
from typing import Sequence

from torch.utils.data.dataloader import default_collate


def get_first_available_dir(dir_list: Sequence[str], strict: bool = True) -> str:
    # looks for the first available dir in a given list
    for d in dir_list:
        if os.path.isdir(d):
            return d

    if strict:
        raise Exception(f"No dir exists in the list: {dir_list}")
    else:
        return ""


def save_pickle(obj, save_path):
    with open(save_path, "wb") as fid:
        pickle.dump(obj, fid)


def load_pickle(save_path):
    with open(save_path, "rb") as fid:
        obj = pickle.load(fid)
    return obj


def my_collate(batch):
    """
    Extend the default collate function to ignore None samples.
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path: str, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def add_str_to_jsonfile(json_file_path: str, item: str):
    items = set()

    if os.path.exists(json_file_path):
        try:
            data = load_json(json_file_path)
            items = set(data)

        except json.JSONDecodeError:
            # In case the file is empty or not a valid JSON, we'll start with an empty set
            pass

    items.add(item)

    save_json(json_file_path, sorted(items))


def normalize_min_max(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor.clamp(0, 1)
    return tensor
