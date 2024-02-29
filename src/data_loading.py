import numpy as np
import os
import random

from PIL import Image
from sklearn.model_selection import train_test_split


class BavarianBuildingDataset:
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = [f for f in os.listdir(image_dir) if 'image' in f]
        self.mask_files = [f for f in os.listdir(mask_dir) if 'umring' in f]

    def load_image(self, image_file):
        img = Image.open(os.path.join(self.image_dir, image_file))
        return np.array(img) / 255.0

    def load_mask(self, mask_file):
        mask = Image.open(os.path.join(self.mask_dir, mask_file))
        return np.array(mask) / 255.0
    
    def get_image_mask_pair(self, index):
        image_file = self.image_files[index]
        mask_file = image_file.replace('image', 'umring')
        return self.load_image(image_file), self.load_mask(mask_file)
    
    def get_batch(self, indices):
        assert 0 <= max(indices) < len(self.image_files), f"Index {max(indices)} out of range"

        images, masks = [], []
        for i in indices:
            image, mask = self.get_image_mask_pair(i)
            images.append(image)
            masks.append(mask)
        return np.stack(images), np.stack(masks)
    
    def subset(self, size):
        assert 0 <= size <= 1, "Subset-size should be a fraction between 0 and 1"

        num_samples = round(len(self.image_files) * size)
        indices = random.sample(range(len(self.image_files)), num_samples)

        subset = BavarianBuildingDataset(self.image_dir, self.mask_dir)
        subset.image_files = [self.image_files[i] for i in indices]
        subset.mask_files = [self.mask_files[i] for i in indices]
        return subset
    
    def train_test_val_split(self, train_size=0.7, test_size=0.15, val_size=0.15, random_state=42):
        assert train_size + test_size + val_size == 1.0, "Split sizes should add up to 1.0"

        total_size = len(self.image_files)
        train_indices, test_indices = train_test_split(range(total_size), test_size=test_size, random_state=random_state)
        train_indices, val_indices = train_test_split(train_indices, test_size=val_size/(train_size+val_size), random_state=random_state)

        return train_indices, test_indices, val_indices
    
    def __len__(self):
        return len(self.image_files)
    

def generator(dataset, indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield dataset.get_batch(indices[i:i + batch_size])
