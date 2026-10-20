import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, allow_nan=True)


def clean_species(value):
    return str(value).strip().lower()


def clean_topology(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def finite_or_nan(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def parse_int_list(value):
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        tokens = value
    else:
        text = str(value).strip()

        if not text or text.lower() in {"none", "null", "all"}:
            return None

        tokens = text.split(",")

    values = []

    for token in tokens:
        token = str(token).strip()

        if not token:
            continue

        values.append(int(token))

    return values or None


def expand_path(value):
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
