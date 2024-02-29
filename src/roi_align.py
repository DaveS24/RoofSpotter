import tensorflow as tf


class ROIAlign:
    def __init__(self, backbone, pool_size=(7, 7), num_rois=32):
        self.backbone = backbone
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.layer = self.build_layer()

    def roi_align(self, inputs): #TODO: Also take the proposed ROIs by the RPN as input
        # Separate the inputs
        feature_map = inputs[0]
        rois = inputs[1]

        # Normalize the coordinates of the ROIs by the spatial dimension of the feature map
        rois_norm = rois / tf.cast(tf.shape(feature_map)[1:3], tf.float32)

        # Use crop_and_resize to extract the ROIs and reduce them to the pool size
        rois_aligned = tf.image.crop_and_resize(feature_map, rois_norm, tf.range(tf.shape(rois)[0]), self.pool_size)

        return rois_aligned

    def build_layer(self):
        # Define the input layers
        feature_map_input = self.backbone.model.output
        rois_input = tf.keras.layers.Input(shape=(self.num_rois, 4))

        # Apply the ROIAlign layer
        x = tf.keras.layers.Lambda(self.roi_align)([feature_map_input, rois_input])

        # Create the layer
        layer = tf.keras.Model(inputs=[feature_map_input, rois_input], outputs=x)
        return layer
