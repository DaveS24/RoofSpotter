import tensorflow as tf

from utils import AnchorGenerator


class RPN:
    def __init__(self, backbone):
        self.backbone = backbone
        self.anchors = AnchorGenerator.generate_anchors() #TODO: Apply the anchors to the feature map and generate region proposals
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
