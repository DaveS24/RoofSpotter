import tensorflow as tf


class MaskHead:
    def __init__(self, config, roi_align, name='Mask_Head'):
        self.config = config
        self.roi_align = roi_align
        
        self.model = self.build_model()
        self.model._name = name

    def build_model(self):
        # Get the aligned ROIs from the ROI Align layer
        roi_aligned = self.roi_align.model.output

        # Apply a series of convolutional and upsampling layers
        x = tf.keras.layers.Conv2D(self.config.mask_head_conv_filters, (3, 3),
                                   padding='same', activation='relu')(roi_aligned)
        x = tf.keras.layers.Conv2D(self.config.mask_head_conv_filters, (3, 3),
                                   padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(self.config.mask_head_upsample_filters, (2, 2),
                                            strides=2, activation='relu')(x)
        
        # Apply the final convolutional layer to predict the binary masks
        binary_masks = tf.keras.layers.Conv2D(self.config.num_classes, (1, 1), activation='sigmoid')(x)

        model = tf.keras.Model(inputs=roi_aligned, outputs=binary_masks)
        return model
    