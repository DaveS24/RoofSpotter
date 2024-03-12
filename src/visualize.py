import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    @staticmethod
    def display_sample(image, mask):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Add the mean pixel value back to each pixel and convert to uint8
        image_display = image + [103.939, 116.779, 123.68]
        image_display = np.clip(image_display, 0, 255).astype('uint8')

        # Convert the images back from BGR to RGB
        image_display = image_display[..., ::-1]

        axes[0].imshow(image_display)
        axes[0].set_title('Original Image')

        # Plot the original mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Original Mask')

        # Overlay the mask on the image
        overlay = np.where(mask[:, :, :3] > 0, [255, 255, 255], image_display)
        axes[2].imshow(overlay)

        # Remove axis labels
        for ax in axes:
            ax.axis('off')

        plt.show()

    @staticmethod
    def display_avg_feature_map(feature_maps):
        averaged_feature_map = np.mean(feature_maps, axis=-1)

        plt.imshow(averaged_feature_map[0], cmap='gray')
        plt.axis('off')
        plt.show()

    @staticmethod
    def display_rois(image, roi_scores, roi_boxes):
        pass
