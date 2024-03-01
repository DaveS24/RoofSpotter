import tensorflow as tf

from utils import AnchorGenerator


class RPN:
    def __init__(self, backbone):
        self.backbone = backbone
        self.anchors = AnchorGenerator.generate_anchors()
        self.num_anchors = len(self.anchors)
        self.model = self.build_model()

    def build_model(self):
        # Get the feature map from the backbone
        feature_map = self.backbone.model.output

        # Apply a 3x3 convolutional layer
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv')(feature_map)
        x = tf.keras.layers.BatchNormalization()(x)

        # Classification layer - 2 outputs per anchor (objectness score)
        x_class = tf.keras.layers.Conv2D(self.num_anchors * 2, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)

        # Regression layer- 4 outputs per anchor (bounding box coordinates)
        x_regr = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regr')(x)

        return tf.keras.Model(inputs=self.backbone.model.input, outputs=[x_class, x_regr])


# TODO:

# Areas for Improvement and Key Missing Elements:

#     Anchor Application and Proposal Generation:  Currently, there's no explicit mechanism to slide the anchors across the feature map and generate object proposals. Here's what you need to add:
#         Sliding Window: Iterate over each spatial location of the feature map.
#         Anchor Mapping: At each location, map the set of anchors (which are likely defined in image coordinates) to the corresponding positions on the feature map. You'll need some calculations considering feature map stride.
#         Proposal Generation: Output the mapped anchors as region proposals.

#     Objectness Scores and Refinement: The RPN needs to both predict whether an anchor likely contains an object and refine the anchor's coordinates to better fit the potential object. You'll need to:
#         Score Calculation: Calculate objectness scores (probabilities) from your classification output (x_class).
#         Bounding Box Regression: Apply the regression adjustments (x_regr) to the anchors.

#     Loss Function:  Implement the multi-task loss function typically used for RPNs. It includes a term for classification (e.g., binary cross-entropy for object/background) and a regression loss (e.g., smooth L1 loss for bounding box coordinates).

#     Non-Maximum Suppression (NMS): The RPN will likely produce many overlapping proposals. You need to implement non-maximum suppression to filter out redundant and lower-scoring proposals.

# Next Steps

#     Anchor-Feature Map Mapping: Write code to map anchors to feature map locations, taking the feature map's stride into account.
#     Proposal Filtering: Add logic to filter out proposals based on objectness scores as well as those extending beyond the image boundaries.
#     Loss Calculation: Implement the RPN loss function. TensorFlow likely contains built-in components for these losses.
#     NMS: Implement non-maximum suppression to refine your proposals.
