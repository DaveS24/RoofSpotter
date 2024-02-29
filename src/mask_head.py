import tensorflow as tf


class MaskHead:
    def __init__(self, roi_align, num_classes=2):
        self.roi_align = roi_align
        self.num_classes = num_classes
        self.layer = self.build_layer()

    def build_layer(self):
        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align.layer.output

        # Apply a series of convolutional and upsampling layers
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(roi_aligned)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(x)
        x = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(x)

        # Create the layer
        layer = tf.keras.Model(inputs=self.roi_align.layer.input, outputs=x)
        return layer
