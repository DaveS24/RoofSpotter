import tensorflow as tf


class Classifier:
    def __init__(self, config, roi_align_layer, name='Classifier'):
        self.config = config
        self.roi_align_layer = roi_align_layer

        self.layer = self.build_layer()
        self.layer._name = name

    def build_layer(self):
        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align_layer.layer.output

        x = tf.keras.layers.Flatten()(roi_aligned)

        # Apply a fully connected layer to extract features
        x = tf.keras.layers.Dense(self.config.classifier_dense_units, activation='relu')(x)

        # Apply a fully connected layer to predict the class scores
        class_scores = tf.keras.layers.Dense(self.config.num_classes, activation='softmax', name='class_scores')(x)

        # Apply a fully connected layer to predict the bounding box coordinates
        bbox = tf.keras.layers.Dense(self.config.num_classes * 4, activation='linear', name='bbox')(x)
        bbox = tf.keras.layers.Reshape((self.config.num_classes, 4))(bbox)

        # Create the layer
        layer = tf.keras.Model(inputs=roi_aligned, outputs=[class_scores, bbox])
        return layer
    
    def compile_model(self):
        self.layer.compile(optimizer=self.config.classifier_optimizer, loss=self._classifier_loss)

    @staticmethod
    def _classifier_loss(y_true, y_pred):
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
