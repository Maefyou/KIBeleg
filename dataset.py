import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class CircleDataset(Dataset):
    """Dataset for circle detection."""
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the directory with images and ground truth.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_name).convert('L')
        
        gt_name = os.path.join(self.data_path, f'gt_{self.image_files[idx].split("_")[1].split(".")[0]}.npy')
        ground_truth = np.load(gt_name, allow_pickle=True)

        # Extract points from the image
        width, height = image.size
        points = []
        for y in range(height):
            for x in range(width):
                points.append((x / width, y / height, image.getpixel((x, y)) / 255.0))
        
        points = torch.tensor(points, dtype=torch.float32)

        vectors = torch.tensor([item['vector'] for item in ground_truth], dtype=torch.float32)
        is_inside = torch.tensor([item['is_inside'] for item in ground_truth], dtype=torch.float32)

        sample = {'points': points, 'vectors': vectors, 'is_inside': is_inside}

        if self.transform:
            sample = self.transform(sample)

        return sample
