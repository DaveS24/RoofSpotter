import tensorflow as tf


class MaskHead:
    def __init__(self, config, roi_align_layer, name='Mask_Head'):
        self.config = config
        self.roi_align_layer = roi_align_layer
        
        self.layer = self.build_layer()
        self.layer._name = name

    def build_layer(self):
        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align_layer.layer.output

        # Apply a series of convolutional and upsampling layers
        x = tf.keras.layers.Conv2D(self.config.mask_head_conv_filters,
                                   self.config.mask_head_conv_kernel_size,
                                   padding='same', activation='relu')(roi_aligned)
        x = tf.keras.layers.Conv2D(self.config.mask_head_conv_filters,
                                   self.config.mask_head_conv_kernel_size,
                                   padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(self.config.mask_head_upsample_filters,
                                            self.config.mask_head_upsample_kernel_size,
                                            strides=2, activation='relu')(x)
        x = tf.keras.layers.Conv2D(self.config.num_classes, (1, 1), activation='sigmoid')(x)

        # Create the layer
        layer = tf.keras.Model(inputs=roi_aligned, outputs=x)
        return layer
    
    def compile_model(self):
        self.layer.compile(optimizer=self.config.mask_head_optimizer, loss=self._mask_loss)

    @staticmethod
    def _mask_loss(y_true, y_pred):
        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Compute the binary cross-entropy loss
        loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

        return loss
