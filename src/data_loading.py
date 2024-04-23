import numpy as np
import os
import random

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split


class BavarianBuildingDataset:
    """
    A dataset class for loading images and masks from the Bavarian Building dataset.

        Attributes:
            image_dir (str): The directory containing the images.
            mask_dir (str): The directory containing the masks.
            image_files (list): The list of image files in the dataset.
            mask_files (list): The list of mask files in the dataset.

        Methods:
            load_image: Load and preprocess the image `image_file` from the dataset.
            load_mask: Load and normalize the mask `mask_file` from the dataset.
            get_image_mask_pair: Get the image-mask pair at the given `index` from the dataset.
            get_batch: Get a batch of images and masks using `indices` from the dataset.
            subset: Create a subset of the dataset with the given `size`.
            train_test_val_split: Split the dataset into training, testing, and validation sets.
            __len__: Get the length of the dataset.
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = [f for f in os.listdir(image_dir) if 'image' in f]
        self.mask_files = [f for f in os.listdir(mask_dir) if 'umring' in f]

    def load_image(self, image_file):
        """
        Load and preprocess the image `image_file` from the dataset.

            Parameters:
                image_file (str): The image file to load.

            Returns:
                img (np.array): The preprocessed image.
        """

        img = Image.open(os.path.join(self.image_dir, image_file))
        img = preprocess_input(np.array(img))
        return img

    def load_mask(self, mask_file):
        """
        Load and normalize the mask `mask_file` from the dataset.

            Parameters:
                mask_file (str): The mask file to load.

            Returns:
                mask (np.array): The normalized mask.
        """

        mask = Image.open(os.path.join(self.mask_dir, mask_file))
        mask = np.array(mask)[:, :, 0] / 255.0
        return mask

    def get_image_mask_pair(self, index):
        """
        Get the image-mask pair at the given `index` from the dataset.

            Parameters:
                index (int): The index of the image-mask pair.

            Returns:
                img (np.array): The preprocessed image.
                mask (np.array): The normalized mask.
        """

        image_file = self.image_files[index]
        mask_file = image_file.replace('image', 'umring')
        return self.load_image(image_file), self.load_mask(mask_file)

    def get_batch(self, indices):
        """
        Get a batch of images and masks using `indices` from the dataset.

            Parameters:
                indices (list): The list of indices for the batch.

            Returns:
                images (np.array): The preprocessed images.
                masks (np.array): The normalized masks.
        """

        assert 0 <= max(indices) < len(self.image_files), f"Index {max(indices)} out of range"

        images, masks = [], []
        for i in indices:
            image, mask = self.get_image_mask_pair(i)
            images.append(image)
            masks.append(mask)
        return np.stack(images), np.stack(masks)

    def subset(self, size=0.1):
        """
        Create a subset of the dataset with the given `size`.

            Parameters:
                size (float): The fraction of the subset.

            Returns:
                subset (BavarianBuildingDataset): The subset of the dataset.
        """

        assert 0 <= size <= 1, "Subset-size should be a fraction between 0 and 1"

        num_samples = round(len(self.image_files) * size)
        indices = random.sample(range(len(self.image_files)), num_samples)

        subset = BavarianBuildingDataset(self.image_dir, self.mask_dir)
        subset.image_files = [self.image_files[i] for i in indices]
        subset.mask_files = [self.mask_files[i] for i in indices]
        return subset

    def train_test_val_split(self, train_size=0.7, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split the dataset into training, testing, and validation sets.

            Parameters:
                train_size (float): The fraction of the training set.
                test_size (float): The fraction of the testing set.
                val_size (float): The fraction of the validation set.
                random_state (int): The random state for reproducibility.

            Returns:
                train_indices (list): The indices for the training set.
                test_indices (list): The indices for the testing set.
                val_indices (list): The indices for the validation set.
        """

        assert train_size + test_size + val_size == 1.0, "Split sizes should add up to 1.0"

        total_size = len(self.image_files)
        train_indices, test_indices = train_test_split(range(total_size), test_size=test_size,
                                                       random_state=random_state)
        train_indices, val_indices = train_test_split(train_indices, test_size=val_size / (train_size + val_size),
                                                      random_state=random_state)

        return train_indices, test_indices, val_indices

    def __len__(self):
        return len(self.image_files)


def generator(dataset, indices, batch_size):
    """
    A generator function to yield batches of images and masks from the dataset.

        Parameters:
            dataset (BavarianBuildingDataset): The dataset to generate batches from.
            indices (list): The list of indices to generate batches from.
            batch_size (int): The batch size.

        Yields:
            images (np.array): The preprocessed images.
            masks (np.array): The normalized masks.
    """

    for i in range(0, len(indices), batch_size):
        yield dataset.get_batch(indices[i:i + batch_size])
