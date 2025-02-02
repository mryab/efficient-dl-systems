import os
import random

import numpy as np
import torch
import pandas as pd


def seed_everything(seed: int = 595959) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Settings:
    batch_size: int = 256
    epochs: int = 2
    lr: float = 3e-5
    gamma: float = 0.7
    seed: int = 42
    device: str = get_device()
    train_frac: float = 0.8


class Clothes:
    directory = "data"
    train_val_img_dir = "train"
    csv_name = "images.csv"
    archive_name = "images_original"


def get_labels_dict():
    frame = pd.read_csv(f"{Clothes.directory}/{Clothes.csv_name}")
    labels = frame["label"].unique()
    return {label: i for i, label in enumerate(labels)}
