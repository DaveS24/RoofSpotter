import tensorflow as tf


class ROIAlignLayer:
    def __init__(self, backbone, rpn, pool_size, name='ROI_Align'):
        self.backbone = backbone
        self.rpn = rpn
        self.pool_size = pool_size
        self.layer = self.build_layer()
        self.layer._name = name

    def build_layer(self):
        # Get the feature map from the backbone
        feature_map = self.backbone.model.output
        # Get the ROIs from the RPN
        _, rois = self.rpn.model.output

        # Normalize the coordinates of the rois
        normalized_rois = rois / tf.constant([self.backbone.model.input_shape[1],
                                              self.backbone.model.input_shape[2],
                                              self.backbone.model.input_shape[1],
                                              self.backbone.model.input_shape[2]], dtype=tf.float32)

        # Use crop_and_resize to perform the ROI Align operation
        roi_align_features = tf.image.crop_and_resize(feature_map, normalized_rois, tf.range(tf.shape(rois)[0]), crop_size=self.pool_size)

        layer = tf.keras.Model(inputs=[feature_map, rois], outputs=roi_align_features)
        return layer
