import numpy as np
import matplotlib.pyplot as plt

DEFAULT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


class Visualizer:
    """
    A class to visualize the results of the Mask R-CNN model and its components.

        Attributes:
            None

        Methods:
            display_sample: Display the original image and mask, and an overlay of the mask on the image.
            display_avg_feature_map: Display the average feature map of the feature maps.
            display_rois: Display the ROIs proposed by the RPN on the original image and the feature map.
            display_aligned_sample: Display one ROI on the feature map and its alignments.
    """

    @classmethod
    def display_sample(cls, image, mask):
        """
        Display the original image and mask, and an overlay of the mask on the image.

            Parameters:
                image (np.array): The original image.
                mask (np.array): The original mask.

            Returns:
                None
        """

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        image = cls._convert_to_original_image(image)

        axes[0].imshow(image)
        axes[0].set_title('Original Image')

        # Plot the original mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Original Mask')

        # Overlay the mask on the image
        mask_3d = np.stack([mask] * 3, axis=-1)
        overlay = np.where(mask_3d > 0, [255, 255, 255], image)
        axes[2].imshow(overlay)

        # Remove axis labels
        for ax in axes:
            ax.axis('off')

        plt.show()

    @staticmethod
    def display_avg_feature_map(feature_map):
        """
        Display the average feature map of the feature map.

            Parameters:
                feature_map (np.array): The feature map.

            Returns:
                None
        """

        averaged_feature_map = np.mean(feature_map, axis=-1)

        plt.imshow(averaged_feature_map, cmap='gray')
        plt.axis('off')
        plt.show()

    @classmethod
    def display_rois(cls, image, feature_map, rois):
        """
        Display the ROIs proposed by the RPN on the original image and the feature map.

            Parameters:
                image (np.array): The original image.
                feature_map (np.array): The feature map.
                rois (np.array): The ROIs proposed by the RPN.

            Returns:
                None
        """

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        image = cls._convert_to_original_image(image)

        axes[0].imshow(image)
        axes[1].imshow(np.mean(feature_map, axis=-1), cmap='gray')

        scale_x = image.shape[0] / (feature_map.shape[0] - 1)
        scale_y = image.shape[1] / (feature_map.shape[1] - 1)

        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = roi
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            # Scale the coordinates to the image shape
            ix1, ix2 = x1 * scale_x, x2 * scale_x
            iy1, iy2 = y1 * scale_y, y2 * scale_y

            axes[0].add_patch(plt.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1, fill=False, edgecolor=color, lw=1))
            axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=1))

        # Remove axis labels
        for ax in axes:
            ax.axis('off')

        plt.show()

    @classmethod
    def display_aligned_rois(cls, feature_map, rois, aligned_rois):
        """
        Display four ROIs on the feature map and its alignments.

            Parameters:
                feature_map (np.array): The feature map.
                rois (np.array): The ROIs proposed by the RPN.
                aligned_rois (np.array): The aligned ROIs.

            Returns:
                None
        """

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # Overlay one ROI on the feature map
        axes[0].imshow(np.mean(feature_map, axis=-1), cmap='gray')
        for i in range(4):
            x1, y1, x2, y2 = rois[i]
            axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            fill=False, edgecolor=DEFAULT_COLORS[i], lw=1))

            # Display the corresponding aligned ROIs
            axes[i + 1].imshow(np.mean(aligned_rois[i], axis=-1), cmap='gray')

        # Remove axis labels
        for ax in axes:
            ax.axis('off')

        plt.show()

    @staticmethod
    def _convert_to_original_image(image):
        """
        Convert the image from BGR to RGB and add the mean pixel value back to each pixel.

        This is done to reverse the preprocessing steps that were applied to the image.

            Parameters:
                image (np.array): The image to convert.

            Returns:
                image_display (np.array): The converted image.
        """

        image_display = image + [103.939, 116.779, 123.68]
        image_display = np.clip(image_display, 0, 255).astype('uint8')

        # Convert the images back from BGR to RGB
        image_display = image_display[..., ::-1]
        return image_display
