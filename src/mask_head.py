import tensorflow as tf


class MaskHead:
    def __init__(self, roi_align_layer, num_classes=2):
        self.roi_align_layer = roi_align_layer
        self.num_classes = num_classes
        self.layer = self.build_layer()

    def build_layer(self):
        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align_layer.layer.output

        # Apply a series of convolutional and upsampling layers
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))(roi_aligned)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu'))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid'))(x)

        # Create the layer
        layer = tf.keras.Model(inputs=self.roi_align_layer.layer.input, outputs=x)
        return layer
    
    def mask_loss(y_true, y_pred):
        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Compute the binary cross-entropy loss
        loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

        return loss
    
    def compile_model(self, optimizer):
        self.layer.compile(optimizer=optimizer, loss=self.mask_loss)


# TODO:
        
# Key Considerations and Areas for Refinement

#     Output Resolution: Consider the desired resolution of your final mask predictions. Do you want them to match the original ROI size or a different resolution? You might need to adjust the number of upsampling or convolutional layers to achieve the correct output shape.

#     Mask Selection: In Mask R-CNN, only the mask associated with the predicted class of an ROI is used during training. You'll need to add logic to:
#         Get the predicted class for each ROI from your classifier.
#         Select or slice the appropriate mask channel from your Mask Head's output.

#     Data Compatibility: Be extra cautious about the dimensions of your masks. Ensure that the ground truth masks (y_true) fed into your loss function align with the output of your Mask Head. There might be some resizing or indexing operations needed.

# Recommendations

#     Debugging: Add print statements or visualizers to inspect the output of your Mask Head. Ensure the shapes align with your ground truth masks, especially after any mask selection operations.
