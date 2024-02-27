import numpy as np
import os

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
        images, masks = [], []
        for i in indices:
            image, mask = self.get_image_mask_pair(i)
            images.append(image)
            masks.append(mask)
        return np.stack(images), np.stack(masks)
    
    def train_test_val_split(self, train_size=0.7, test_size=0.15, val_size=0.15, random_state=42):
        assert train_size + test_size + val_size == 1.0, "Split sizes should add up to 1.0"

        total_size = len(self.image_files)
        train_indices, test_indices = train_test_split(range(total_size), test_size=test_size, random_state=random_state)
        train_indices, val_indices = train_test_split(train_indices, test_size=val_size/(train_size+val_size), random_state=random_state)

        return train_indices, test_indices, val_indices
    
    def __len__(self):
        return len(self.image_files)
    

def generator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset.get_batch(range(i, min(i + batch_size, len(dataset))))
