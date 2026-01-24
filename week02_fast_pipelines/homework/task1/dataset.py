import os
from os.path import isfile, join
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Carvana(Dataset):
    def __init__(self, root: str, transform: transforms.Compose = None) -> None:
        """
        :param root: path to the data folder
        :param transform: transforms of the images and labels
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        (self.data_path, self.labels_path) = ([], [])

        def load_images(path: str) -> List[str]:
            """
            Return a list with paths to all images

            :param path: path to the data folder
            :return: list with paths to all images
            """
            images_dir = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
            images_dir.sort()

            return images_dir

        self.data_path = load_images(self.root + "/train")
        self.labels_path = load_images(self.root + "/train_masks")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param index: sample index
        :return: tuple (img, target) with the input data and its label
        """
        img = Image.open(self.data_path[index])
        target = Image.open(self.labels_path[index])

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
            target = (target > 0).float()

        return img, target

    def __len__(self):
        return len(self.data_path)


def get_train_data() -> torch.utils.data.DataLoader:
    train_dataset = Carvana(
        root=".", transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4
    )

    return train_loader


def im_show(img_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    """
    Plots images with corresponding segmentation masks

    :param img_list: list of pairs image-mask
    """
    to_PIL = transforms.ToPILImage()
    if len(img_list) > 9:
        raise Exception("len(img_list) must be smaller than 10")
    fig, axes = plt.subplots(len(img_list), 2, figsize=(16, 16))
    fig.tight_layout()

    for (idx, sample) in enumerate(img_list):
        axes[idx][0].imshow(np.array(to_PIL(sample[0])))
        axes[idx][1].imshow(np.array(to_PIL(sample[1])))
        for ax in axes[idx]:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()
