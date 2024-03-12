import tensorflow as tf

from utils import AnchorGenerator


class RPN:
    def __init__(self, config, backbone, name='RPN'):
        self.config = config
        self.backbone = backbone

        self.model = self.build_model()
        self.model._name = name

    def build_model(self):
        # Get the feature map from the backbone
        feature_maps = self.backbone.model.output
        feature_maps_shape = feature_maps.shape

        num_anchors = len(self.config.anchor_scales) * len(self.config.anchor_ratios)

        # Apply a convolutional layer to the feature map
        conv_layer = tf.keras.layers.Conv2D(self.config.rpn_conv_filters,
                                            self.config.rpn_conv_kernel_size,
                                            padding='same', activation='relu')(feature_maps)

        # Apply a 1x1 convolutional layer to determine whether each anchor contains an object or not
        roi_scores = tf.keras.layers.Conv2D(num_anchors * 2, (1, 1), activation='sigmoid')(conv_layer)

        # Apply a 1x1 convolutional layer to refine the bounding box coordinates for each anchor
        roi_boxes = tf.keras.layers.Conv2D(num_anchors * 4, (1, 1))(conv_layer)
        
        # Transform the roi_boxes to the absolute coordinates in the original image
        anchors = AnchorGenerator.generate_anchors(self.config, feature_map_width=feature_maps_shape[2], feature_map_height=feature_maps_shape[1])
        roi_boxes = ...

        # Apply Non-Maximum Suppression (NMS) to the proposed regions
        roi_scores, roi_boxes = self._non_max_suppression(roi_scores, roi_boxes)

        model = tf.keras.Model(inputs=feature_maps, outputs=[roi_scores, roi_boxes])
        return model
    
    def _non_max_suppression(self, roi_scores, roi_boxes):
        roi_boxes, roi_scores = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(roi_boxes, [tf.shape(roi_boxes)[0], -1, 4]),
            scores=tf.reshape(roi_scores, [tf.shape(roi_scores)[0], -1]),
            max_output_size_per_class=self.config.rpn_max_proposals,
            max_total_size=self.config.rpn_max_proposals,
            iou_threshold=self.config.rpn_iou_threshod,
            score_threshold=self.config.rpn_score_threshold
        )
        return roi_scores, roi_boxes


# TODO: Solve this mess 

class DecodePredictions(tf.keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        super(DecodePredictions, self).__init__(**kwargs)
        self.anchors = anchors

    def call(self, inputs):
        roi_scores, roi_boxes = inputs

        # Convert anchors to the center-width-height format
        anchor_x = (self.anchors[:, 2] + self.anchors[:, 0]) / 2
        anchor_y = (self.anchors[:, 3] + self.anchors[:, 1]) / 2
        anchor_w = self.anchors[:, 2] - self.anchors[:, 0]
        anchor_h = self.anchors[:, 3] - self.anchors[:, 1]

        # Apply the predicted offsets to the anchors
        roi_x = roi_boxes[..., 0] * anchor_w + anchor_x
        roi_y = roi_boxes[..., 1] * anchor_h + anchor_y
        roi_w = tf.exp(roi_boxes[..., 2]) * anchor_w
        roi_h = tf.exp(roi_boxes[..., 3]) * anchor_h

        # Convert back to the top-left, bottom-right format
        roi_boxes = tf.stack([roi_x - roi_w / 2,
                              roi_y - roi_h / 2,
                              roi_x + roi_w / 2,
                              roi_y + roi_h / 2], axis=-1)

        return [roi_scores, roi_boxes]
