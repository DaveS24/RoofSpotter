import tensorflow as tf


class Classifier:
    def __init__(self, roi_align_layer, num_classes=2):
        self.roi_align_layer = roi_align_layer
        self.num_classes = num_classes
        self.layer = self.build_layer()

    def build_layer(self):
        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align_layer.layer.output

        # Apply a convolutional layer to reduce dimensionality
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (2, 2), activation='relu'))(roi_aligned)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

        # Apply a fully connected layer to extract features
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        # Apply a fully connected layer to predict the class scores
        class_scores = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='class_scores')(x)

        # Apply a fully connected layer to predict the bounding box coordinates
        bbox = tf.keras.layers.Dense(self.num_classes * 4, activation='linear', name='bbox')(x)
        bbox = tf.keras.layers.Reshape((-1, self.num_classes, 4))(bbox)

        # Create the layer
        layer = tf.keras.Model(inputs=self.roi_align_layer.layer.input, outputs=[class_scores, bbox])
        return layer
    
    def classifier_loss(y_true, y_pred):
        # Separate the ground truth labels and bounding boxes
        y_true_class = y_true[:, :, :1]
        y_true_bbox = y_true[:, :, 1:]

        # Separate the predicted class scores and bounding boxes
        y_pred_class = y_pred[:, :, :1]
        y_pred_bbox = y_pred[:, :, 1:]

        # Compute the classification loss
        classification_loss = tf.keras.losses.categorical_crossentropy(y_true_class, y_pred_class)

        # Compute the bounding box regression loss
        regression_loss = tf.keras.losses.huber(y_true_bbox, y_pred_bbox)

        # Combine the losses
        total_loss = classification_loss + regression_loss

        return total_loss
    
    def compile_model(self, optimizer):
        self.layer.compile(optimizer=optimizer, loss=self.classifier_loss)


# TODO:
        
# Key Considerations and Areas for Refinement

#     Bounding Box Coordinate Representation:
#         Ensure that the way you represent bounding boxes in your ground truth (y_true_bbox) and predictions (y_pred_bbox) is consistent. Are they relative offsets to the ROIs, or absolute image coordinates? Your loss calculation needs to align with this representation.

#     Masking:
#         Currently, your loss function processes all bounding box predictions. In Mask R-CNN, typically, only the bounding box adjustments associated with the predicted class of the ROI are used during training. You'll likely need some masking or selection operations within your loss function to achieve this.

# Recommendations

#     Input/Output Compatibility: Make sure the output of your ROI Align layer is correctly formatted as input to your classifier.
#     Debugging and Visualization: Consider using print statements or visualizers within your loss function to inspect the alignment between ground truth and predicted bounding boxes.
