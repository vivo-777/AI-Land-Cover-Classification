"""
PyTorch Dataset for Delhi NCR Land Cover Classification
========================================================
Custom Dataset that loads satellite RGB patches and returns
(image_tensor, label_id) pairs for training / evaluation.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CATEGORY_TO_ID, RGB_DIR


# Default transforms: convert PIL image → tensor (scales to [0, 1])
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
])


class DelhiLandCoverDataset(Dataset):
    """
    Dataset for loading satellite image patches with land cover labels.

    Args:
        image_filenames: list of image file names (e.g., '28.2056_76.8558.png')
        labels:          list of category strings (e.g., 'Built-up', 'Cropland')
        image_dir:       path to the directory containing the PNG images
        transform:       torchvision transforms to apply to each image
    """

    def __init__(
        self,
        image_filenames: list[str],
        labels: list[str],
        image_dir: str = RGB_DIR,
        transform=None,
    ):
        self.image_filenames = image_filenames
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform or DEFAULT_TRANSFORM

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        img_name = self.image_filenames[idx]
        label_text = self.labels[idx]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_id = CATEGORY_TO_ID[label_text]
        return image, label_id
