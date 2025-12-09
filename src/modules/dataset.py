import cv2
import json
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
import torch
import numpy as np

class DewrapDataset(Dataset):
    def __init__(
        self,
        images_names: List,
        coords: Optional[List] = None,
        transforms=None,
    ):
        self.images_names = images_names
        self.coords = coords
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple:
        # Read image
        image_path = self.images_names[idx]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Apply augs
        if self.coords is not None:
            coords = json.loads(self.coords[idx])
            coords = [[int(round(width * x)),
                       int(round(height * y))
                       ] for x, y in coords]
            transformed = self.transforms(image=image, keypoints=coords)
            image = transformed['image'].float()
            h, w = image.shape[1:]
            coords = transformed['keypoints']
            coords = [[x / w, y / h] for (x, y) in coords]
            coords = torch.tensor(coords).view(-1)
            return image, image_path, (height, width), coords
        else:
            transformed = self.transforms(image=image)
            image = transformed['image'].float()
            return image, image_path, (height, width), image.shape[1:]

    def __len__(self) -> int:
        return len(self.images_names)
