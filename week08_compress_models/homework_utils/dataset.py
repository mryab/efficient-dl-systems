from typing import Callable, Dict, Optional

import pathlib
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset


class AlignedDataset(Dataset):
    
    def __init__(
        self,
        root: str,
        is_train: bool = True,
        transform: Optional[Callable] = None,
        direction: str = 'AtoB'
    ):
        self._root = root
        self._is_train = is_train
        self._transform = transform
        self._direction = direction
        
        self._images_path = pathlib.Path(root) / ('train' if is_train else 'val')
        self._paths = list(self._images_path.rglob('*.jpg'))
    
    def __len__(self) -> int:
        return len(self._paths)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self._paths[index]
        image = Image.open(path).convert('RGB')
        
        w, h = image.size
        w2 = int(w / 2)
        image_A = np.array(image.crop((0, 0, w2, h)))
        image_B = np.array(image.crop((w2, 0, w, h)))
        
        if self._transform is not None:
            result = self._transform(image=image_A, mask=image_B)
            image_A, image_B = result['image'], result['mask']

        if self._direction == 'BtoA':
            image_A, image_B = image_B, image_A
        return image_A, image_B
        
