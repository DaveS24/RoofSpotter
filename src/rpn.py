import numpy as np
import tensorflow as tf


class RPN:
    def __init__(self, config, backbone, name='RPN'):
        self.config = config
        self.backbone = backbone

        self.model = self.build_model()
        self.model._name = name

    def build_model(self):
        # Get the feature maps from the backbone
        feature_maps = self.backbone.model.output

        anchors = self.generate_anchors() # [width, height], shape: (64, 2)
        num_anchors = anchors.shape[0]

        # Reduce depth of the feature map
        shared_layer = tf.keras.layers.Conv2D(self.config.rpn_conv_filters, (3, 3),
                                              padding='same', activation='relu')(feature_maps)

        # Bounding box regression offsets for each anchor
        # (dx, dy, dw, dh), (dx, dy, dw, dh), ... for all 64 anchors, shape: (None, 8, 8, 256)
        pred_anchor_offsets = tf.keras.layers.Conv2D(num_anchors * 4, (1, 1))(shared_layer)

        # Score whether an anchor contains an object or not
        # (roof_score, no_roof_score), (roof_score, no_roof_score) ... for all 64 anchors, shape: (None, 8, 8, 128)
        roi_scores = tf.keras.layers.Conv2D(num_anchors * 2, (1, 1), activation='sigmoid')(shared_layer)

        # Apply Non-Maximum Suppression (NMS) to the proposed offsets
        filtered_anchor_offsets = self.non_maximum_suppression(pred_anchor_offsets, roi_scores)

        # Transform the predicted offsets to absolute coordinates in the feature maps
        roi_boxes = self.decode_offsets(filtered_anchor_offsets, anchors)

        model = tf.keras.Model(inputs=feature_maps, outputs=roi_boxes)
        return model
    
    def generate_anchors(self):
        scales = self.config.anchor_scales # [0.5, 1, 1.5, 2]
        ratios = self.config.anchor_ratios # [1, 1.5, 2]

        lengths = np.outer(scales, ratios).ravel()
        lengths = np.sort(np.unique(lengths)) # [0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 3.0, 4.0]

        anchors = np.array([[width, height] for width in lengths for height in lengths])
        return anchors
    
    def non_maximum_suppression(self, offsets, roi_scores):
        reshaped_offsets = tf.reshape(offsets, (-1, 4)) # (None, 8, 8, 256) -> (None * 64, 4)
        reshaped_scores = tf.reshape(roi_scores, (-1, 2)) # (None, 8, 8, 128) -> (None * 64, 2)
        reshaped_scores = reshaped_scores[:, 0] # Only use true scores

        selected_indices = tf.image.non_max_suppression(reshaped_offsets, reshaped_scores,
                                                        max_output_size=self.config.rpn_max_proposals,
                                                        iou_threshold=self.config.rpn_iou_threshold,
                                                        score_threshold=self.config.rpn_score_threshold)

        filtered_offsets = tf.gather(reshaped_offsets, selected_indices)
        filtered_offsets = tf.reshape(filtered_offsets, tf.shape(offsets)) # Reshape back to original shape
        return filtered_offsets
    
    def decode_offsets(self, offsets, anchors):
        exit()
