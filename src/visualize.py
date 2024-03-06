import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    @staticmethod
    def display_sample(image, mask):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')

        # Plot the original mask
        axes[1].imshow(mask)
        axes[1].set_title('Original Mask')

        # Overlay the mask on the image
        overlay = np.where(mask[:, :, :3] > 0, [1, 1, 1], image)
        axes[2].imshow(overlay)

        # Remove axis labels
        for ax in axes:
            ax.axis('off')

        plt.show()

    @staticmethod
    def display_feature_maps(backbone, image):
        # Get the feature maps from the backbone
        feature_maps = backbone.model.predict(np.expand_dims(image, axis=0))[0]

        # Get the number of feature maps
        num_feature_maps = feature_maps.shape[-1]

        # Create a grid of images
        fig, axes = plt.subplots((num_feature_maps + 15) // 16, 16, figsize=(15, 15))

        # Plot each feature map
        for i, ax in enumerate(axes.flat):
            if i < num_feature_maps:
                ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.axis('off')

        plt.show()
