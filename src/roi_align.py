import tensorflow as tf


class ROIAlignLayer:
    def __init__(self, backbone, pool_size=(7, 7), num_rois=32):
        self.backbone = backbone
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.layer = self.build_layer()

    def roi_align(self, inputs):
        # Separate the inputs
        feature_map = inputs[0]
        rois = inputs[1]
        rpn_proposals = inputs[2]

        # Ensure the RPN proposals are in the correct format [y1, x1, y2, x2]
        rpn_proposals = tf.stack([
            rpn_proposals[:, 1],  # y1
            rpn_proposals[:, 0],  # x1
            rpn_proposals[:, 3],  # y2
            rpn_proposals[:, 2]   # x2
        ], axis=-1)

        # Combine the ROIs and RPN proposals
        combined_rois = tf.concat([rois, rpn_proposals], axis=0)

        # Normalize the coordinates of the ROIs by the spatial dimension of the feature map
        rois_norm = combined_rois / tf.cast(tf.shape(feature_map)[1:3], tf.float32)

        # Use crop_and_resize to extract the ROIs and reduce them to the pool size
        rois_aligned = tf.image.crop_and_resize(feature_map, rois_norm, tf.range(tf.shape(combined_rois)[0]), self.pool_size)

        return rois_aligned

    def build_layer(self):
        # Define the input layers
        feature_map_input = self.backbone.model.output
        rois_input = tf.keras.layers.Input(shape=(self.num_rois, 4))
        rpn_proposals_input = tf.keras.layers.Input(shape=(None, 4))

        # Apply the ROIAlign layer
        x = tf.keras.layers.Lambda(self.roi_align)([feature_map_input, rois_input, rpn_proposals_input])

        # Create the layer
        layer = tf.keras.Model(inputs=[feature_map_input, rois_input, rpn_proposals_input], outputs=x)
        return layer
