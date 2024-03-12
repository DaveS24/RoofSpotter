import tensorflow as tf


class ROIAlignLayer:
    def __init__(self, config, backbone, rpn, name='ROI_Align'):
        self.config = config
        self.backbone = backbone
        self.rpn = rpn

        self.layer = self.build_layer()
        self.layer._name = name

    def build_layer(self):
        # Get the feature map from the backbone
        feature_map = self.backbone.model.output
        # Get the ROI boxes from the RPN
        _, roi_boxes = self.rpn.model.output

        # Normalize the coordinates of the rois
        normalized_rois = roi_boxes / tf.constant([self.backbone.model.input_shape[1],
                                                   self.backbone.model.input_shape[2],
                                                   self.backbone.model.input_shape[1],
                                                   self.backbone.model.input_shape[2]], dtype=tf.float32)

        # Use crop_and_resize to perform the ROI Align operation
        roi_align_features = tf.image.crop_and_resize(feature_map, normalized_rois, tf.range(tf.shape(roi_boxes)[0]), crop_size=self.config.roi_align_pool_size)

        layer = tf.keras.Model(inputs=[feature_map, roi_boxes], outputs=roi_align_features)
        return layer
