import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class CircleDataset(Dataset):
    """Dataset for circle detection."""
    def __init__(self, data_path, transform=None, target_width=128, target_height=128):
        """
        Args:
            data_path (string): Path to the directory with images and ground truth.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_width (int): The target width to resize images to.
            target_height (int): The target height to resize images to.
        """
        self.data_path = data_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.png')]
        self.target_width = target_width
        self.target_height = target_height

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_name).convert('L')
        original_width, original_height = image.size
        
        # Resize image
        image = image.resize((self.target_width, self.target_height), Image.BILINEAR)
        
        gt_name = os.path.join(self.data_path, f'gt_{self.image_files[idx].split("_")[1].split(".")[0]}.npy')
        ground_truth = np.load(gt_name, allow_pickle=True)

        # Extract points from the image
        width, height = image.size
        points = []
        for y in range(height):
            for x in range(width):
                points.append((x / width, y / height, image.getpixel((x, y)) / 255.0))
        
        points = torch.tensor(points, dtype=torch.float32)

        # Scale ground truth vectors
        scale_x = self.target_width / original_width
        scale_y = self.target_height / original_height
        
        vectors = torch.tensor([item['vector'] for item in ground_truth], dtype=torch.float32)
        vectors[:, 0] *= scale_x
        vectors[:, 1] *= scale_y
        
        is_inside = torch.tensor([item['is_inside'] for item in ground_truth], dtype=torch.float32)

        # Since the image is resized, we need to re-evaluate which points are inside the circle
        # For simplicity, we'll just use the original is_inside values, but a more accurate approach
        # would involve scaling the circle parameters and re-calculating the mask.
        # This is a limitation of the current approach.
        if len(is_inside) != self.target_width * self.target_height:
            # This is a simple and not very accurate way to handle resizing of the mask
            is_inside = torch.from_numpy(np.array(Image.fromarray(is_inside.numpy().reshape(original_height, original_width)).resize((self.target_width, self.target_height), Image.NEAREST)).flatten()).float()


        sample = {'points': points, 'vectors': vectors, 'is_inside': is_inside}

        if self.transform:
            sample = self.transform(sample)

        return sample
