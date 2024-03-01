import tensorflow as tf


class Classifier:
    def __init__(self, roi_align_layer, num_classes=2):
        self.roi_align_layer = roi_align_layer
        self.num_classes = num_classes
        self.layer = self.build_layer()

    def build_layer(self):
        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align_layer.layer.output
        x = tf.keras.layers.Flatten()(roi_aligned)

        # Apply a fully connected layer to extract features
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        # Apply a fully connected layer to predict the class scores
        class_scores = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='class_scores')(x)
        # Apply a fully connected layer to predict the bounding box coordinates
        bbox = tf.keras.layers.Dense(self.num_classes * 4, activation='linear', name='bbox')(x)

        # Create the layer
        layer = tf.keras.Model(inputs=self.roi_align_layer.layer.input, outputs=[class_scores, bbox])
        return layer


# TODO:
    
# Key Considerations and Missing Elements

#     Bounding Box Representation:  Think carefully about how bounding box coordinates are represented in your bbox output. Ensure you understand whether the adjustments are relative to the ROIs or absolute coordinates in the image space.

#     Loss Function: Similar to the RPN, you'll need a multi-task loss for the classifier and regressor. It will likely include classification loss (e.g., cross-entropy) and a bounding box regression loss (e.g., smooth L1 loss).

#     Class-Specific Bounding Boxes:  Typically, in Mask R-CNN, the bounding box regression output has 4 values per class, allowing the model to fine-tune bounding boxes independently for each class. You might need to reshape your bbox output accordingly.

# Next Steps

#     Clarify Coordinate Representation: Check if your bounding box outputs require transformation (from relative to absolute coordinates, or vice versa).
#     Reshape Bounding Box Output: If necessary, modify your code to generate class-specific bounding box regression adjustments.
#     Implement Loss Function: Implement the combined classification and regression loss.

# Important Note: Consider how the outputs of your classifier and bounding box regressor will be used to select the final predicted boxes and masks. There's often a step of filtering or merging based on class scores and the adjusted bounding boxes.
