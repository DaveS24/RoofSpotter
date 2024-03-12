import numpy as np
import tensorflow as tf


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

        anchors = AnchorGenerator.generate_anchors(self.config, fm_height=feature_maps_shape[1], fm_width=feature_maps_shape[2])
        num_anchors = len(self.config.anchor_ratios) * len(self.config.anchor_scales)

        # Reduce depth of the feature map
        shared_layer = tf.keras.layers.Conv2D(self.config.rpn_conv_filters, (3, 3),
                                              padding='same', activation='relu')(feature_maps)

        # Bounding box regression offsets for each anchor
        roi_boxes = tf.keras.layers.Conv2D(num_anchors * 4, (1, 1))(shared_layer)
        roi_boxes = tf.reshape(roi_boxes, [feature_maps_shape[1] * feature_maps_shape[2] * num_anchors, 4])

        # Score whether an anchor contains an object or not
        roi_scores = tf.keras.layers.Conv2D(num_anchors * 2, (1, 1), activation='sigmoid')(shared_layer)
        roi_scores = tf.reshape(roi_scores, [feature_maps_shape[1] * feature_maps_shape[2] * num_anchors, 2])

        # Decode the bounding box regression offsets to the absolute image coordinates
        roi_boxes = AnchorGenerator.decode_boxes(roi_boxes, anchors)
        roi_boxes = self.clip_boxes(roi_boxes)
        roi_boxes, roi_scores = self.remove_zero_area_boxes(roi_boxes, roi_scores)

        # Apply Non-Maximum Suppression (NMS) to the proposed regions
        roi_boxes, roi_scores = self.non_maximum_suppression(roi_boxes, roi_scores)

        model = tf.keras.Model(inputs=feature_maps, outputs=[roi_boxes, roi_scores])
        return model
    
    def clip_boxes(self, roi_boxes):
        image_shape = self.config.image_shape
        x1 = tf.clip_by_value(roi_boxes[:, 0], 0, image_shape[1] - 1)
        y1 = tf.clip_by_value(roi_boxes[:, 1], 0, image_shape[0] - 1)
        x2 = tf.clip_by_value(roi_boxes[:, 2], 0, image_shape[1] - 1)
        y2 = tf.clip_by_value(roi_boxes[:, 3], 0, image_shape[0] - 1)

        return tf.stack([x1, y1, x2, y2], axis=1)
    
    def remove_zero_area_boxes(self, roi_boxes, roi_scores):
        width = roi_boxes[:, 2] - roi_boxes[:, 0]
        height = roi_boxes[:, 3] - roi_boxes[:, 1]

        # Create a mask for boxes with non-zero width and height
        mask = tf.logical_and(width > 0, height > 0)

        roi_boxes = tf.boolean_mask(roi_boxes, mask)
        roi_scores = tf.boolean_mask(roi_scores, mask)

        return roi_boxes, roi_scores
    
    def non_maximum_suppression(self, roi_boxes, roi_scores):
        obj_scores = roi_scores[:, 0]
        obj_scores = tf.reshape(obj_scores, [-1])

        selected_indices = tf.image.non_max_suppression(roi_boxes, obj_scores,
                                                        max_output_size=self.config.rpn_max_proposals,
                                                        iou_threshold=self.config.rpn_iou_threshold,
                                                        score_threshold=self.config.rpn_score_threshold)

        selected_boxes = tf.gather(roi_boxes, selected_indices)
        selected_scores = tf.gather(roi_scores, selected_indices)

        return selected_boxes, selected_scores


class AnchorGenerator:
    @classmethod
    def generate_anchors(cls, config, fm_height, fm_width):
        '''Generates anchors in the [x1, y1, x2, y2] format for each feature map cell using the anchor base size, ratios, and scales.'''
        base_size = config.anchor_base_size
        ratios = config.anchor_ratios
        scales = config.anchor_scales

        all_anchors = []

        # Iterate over the feature map
        for y in range(fm_height):
            for x in range(fm_width):
                # Generate the base anchor at the current location
                base_anchor = np.array([x, y, x + base_size, y + base_size])

                # Generate ratio anchors from the base anchor
                ratio_anchors = cls._ratio_enum(base_anchor, ratios)

                # Generate scale anchors from the ratio anchors
                anchors = np.vstack([cls._scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
                all_anchors.append(anchors)

        all_anchors = np.vstack(all_anchors)
        return tf.convert_to_tensor(all_anchors, dtype=tf.float32)
    
    @classmethod
    def _ratio_enum(cls, anchor, ratios):
        w, h, x_ctr, y_ctr = cls._ctr_and_size(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = cls._make_anchors(ws, hs, x_ctr, y_ctr)
        return anchors

    @classmethod
    def _scale_enum(cls, anchor, scales):
        w, h, x_ctr, y_ctr = cls._ctr_and_size(anchor)
        ws = np.array(w) * scales
        hs = np.array(h) * scales
        anchors = cls._make_anchors(ws, hs, x_ctr, y_ctr)
        return anchors
    
    @staticmethod
    def _ctr_and_size(anchor):
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    @staticmethod
    def _make_anchors(ws, hs, x_ctr, y_ctr):
        anchors = np.vstack([x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)]).transpose()
        return anchors
    
    @classmethod
    def decode_boxes(cls, roi_boxes, anchors):
        # Compute the width, height, and center of the anchor boxes
        w = anchors[:, 2] - anchors[:, 0]
        h = anchors[:, 3] - anchors[:, 1]
        x_center = anchors[:, 0] + 0.5 * w
        y_center = anchors[:, 1] + 0.5 * h

        # Apply the predicted offsets
        x_center += roi_boxes[:, 0] * w
        y_center += roi_boxes[:, 1] * h
        w *= tf.exp(roi_boxes[:, 2])
        h *= tf.exp(roi_boxes[:, 3])

        # Convert the width, height, and center to the top-left and bottom-right coordinates
        x1 = x_center - 0.5 * w
        y1 = y_center - 0.5 * h
        x2 = x_center + 0.5 * w
        y2 = y_center + 0.5 * h

        # Concatenate the coordinates to get the proposed regions
        proposed_regions = tf.stack([x1, y1, x2, y2], axis=1)

        return proposed_regions
