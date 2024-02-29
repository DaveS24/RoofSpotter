import tensorflow as tf


class Classifier:
    def __init__(self, roi_align, num_classes=2):
        self.roi_align = roi_align
        self.num_classes = num_classes
        self.layer = self.build_layer()

    def build_layer(self):
        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align.layer.output
        x = tf.keras.layers.Flatten()(roi_aligned)

        # Apply a fully connected layer to extract features
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        # Apply a fully connected layer to predict the class scores
        class_scores = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='class_scores')(x)
        # Apply a fully connected layer to predict the bounding box coordinates
        bbox = tf.keras.layers.Dense(self.num_classes * 4, activation='linear', name='bbox')(x)

        # Create the layer
        layer = tf.keras.Model(inputs=self.roi_align.layer.input, outputs=[class_scores, bbox])
        return layer
