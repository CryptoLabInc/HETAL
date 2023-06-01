from pathlib import Path
from typing import Tuple

import numpy as np


DATA_LIST = [
    "snips",
    "mnist",
    "cifar10",
    "dermamnist",
    "face-mask-detection",
]

DATA_NUM_CLASSES = {
    "snips": 7,
    "mnist": 10,
    "cifar10": 10,
    "dermamnist": 7,
    "face-mask-detection": 3,
}

file_dir = Path(__file__).resolve().parent


def load_data(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    data_dir = file_dir / f"data/{name}"
    X_train = np.load(data_dir / "features_train.npy")
    X_val = np.load(data_dir / "features_val.npy")
    X_test = np.load(data_dir / "features_test.npy")
    y_train = np.load(data_dir / "labels_train.npy")
    y_val = np.load(data_dir / "labels_val.npy")
    y_test = np.load(data_dir / "labels_test.npy")
    num_classes = DATA_NUM_CLASSES[name]
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes
