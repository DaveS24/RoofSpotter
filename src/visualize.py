import numpy as np
import matplotlib.pyplot as plt


DEFAULT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


class Visualizer:
    @classmethod
    def display_sample(cls, image, mask):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        image = cls._convert_to_original_image(image)

        axes[0].imshow(image)
        axes[0].set_title('Original Image')

        # Plot the original mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Original Mask')

        # Overlay the mask on the image
        overlay = np.where(mask[:, :, :3] > 0, [255, 255, 255], image)
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

    @classmethod
    def display_rois(cls, image, feature_maps, rois):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        image = cls._convert_to_original_image(image)

        axes[0].imshow(image)
        axes[1].imshow(np.mean(feature_maps, axis=-1)[0], cmap='gray')

        scale_x = image.shape[0] / feature_maps.shape[1]
        scale_y = image.shape[1] / feature_maps.shape[2]

        fm_offset = -0.5 # Offset to align the ROIs with the feature map due to a shift that plt produces

        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = roi
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            # Scale the coordinates to the image shape
            ix1, ix2 = x1 * scale_x, x2 * scale_x
            iy1, iy2 = y1 * scale_y, y2 * scale_y

            # Apply the offset
            x1, y1, x2, y2 = x1 + fm_offset, y1 + fm_offset, x2 + fm_offset, y2 + fm_offset

            axes[0].add_patch(plt.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1, fill=False, edgecolor=color, lw=1))
            axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=1))

        # Remove axis labels
        for ax in axes:
            ax.axis('off')

        plt.show()

    @staticmethod
    def _convert_to_original_image(image):
        '''Convert the image from BGR to RGB and add the mean pixel value back to each pixel.
        This is done to reverse the preprocessing steps that were applied to the image.'''
        image_display = image + [103.939, 116.779, 123.68]
        image_display = np.clip(image_display, 0, 255).astype('uint8')

        # Convert the images back from BGR to RGB
        image_display = image_display[..., ::-1]
        return image_display
