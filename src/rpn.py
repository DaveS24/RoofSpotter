import numpy as np
import tensorflow as tf


class RPN:
    '''
    The Region Proposal Network (RPN) for the Mask R-CNN model.
    
        Attributes:
            config (Config): The configuration settings.
            backbone (Backbone): The backbone for the Mask R-CNN model.
            model (tf.keras.Model): The RPN model.
        
        Methods:
            build_model: Link the components to build the RPN model.
            generate_anchors: Generate the anchors for the RPN to predict offsets for.
            decode_offsets: Transform the predicted offsets to absolute coordinates in the feature maps.
            clip_boxes: Clip the predicted coordinates that are outside the feature maps.
            non_maximum_suppression: Apply Non-Maximum Suppression (NMS) to the proposed coordinates.
    '''

    def __init__(self, config, backbone, name='RPN'):
        self.config = config
        self.backbone = backbone

        self.model = self.build_model()
        self.model._name = name


    def build_model(self):
        '''
        Link the components to build the RPN model.

            Parameters:
                None

            Returns:
                model (tf.keras.Model): The RPN model.
        '''

        input_feature_maps = tf.keras.layers.Input(shape=self.backbone.model.output.shape[1:], batch_size=self.config.batch_size,
                                                   name='input_feature_maps')
        feature_maps_shape = input_feature_maps.shape

        anchors = self.generate_anchors() # [width, height], Shape: (64, 2)
        num_anchors = anchors.shape[0]

        # Reduce depth of the feature map
        shared_layer = tf.keras.layers.Conv2D(self.config.rpn_conv_filters, (3, 3),
                                              padding='same', activation='relu')(input_feature_maps)

        # Bounding box regression offsets for each anchor
        # (dx, dy, dw, dh), (dx, dy, dw, dh), ... for all 64 anchors, Shape: (batch_size, 8, 8, 256)
        pred_anchor_offsets = tf.keras.layers.Conv2D(num_anchors * 4, (1, 1))(shared_layer)

        # Score whether an anchor contains an object or not
        # (obj_score), (obj_score) ... for all 64 anchors, Shape: (batch_size, 8, 8, 64)
        roi_scores = tf.keras.layers.Conv2D(num_anchors, (1, 1), activation='sigmoid')(shared_layer)

        # Transform the predicted offsets to absolute coordinates in the feature maps, Shape: (batch_size, 8*8*64, 4)
        pred_decoded = self.decode_offsets(pred_anchor_offsets, anchors, feature_maps_shape)

        # Clip or remove the predicted coordinates that are outside the feature maps
        clipped_boxes = self.clip_boxes(pred_decoded, feature_maps_shape) # TODO: How will this impact the loss function? Remove zero-area boxes?

        # Apply Non-Maximum Suppression (NMS) to the proposed coordinates, Shape: (batch_size, num_selected, 4)
        roi_boxes = self.non_maximum_suppression(clipped_boxes, roi_scores)

        model = tf.keras.Model(inputs=input_feature_maps, outputs=roi_boxes)
        return model
    
    
    def generate_anchors(self):
        '''
        Generate the anchors for the RPN to predict offsets for.

            Parameters:
                None

            Returns:
                anchors (np.array): The generated anchors for the RPN to predict offsets for.
        '''

        scales = self.config.anchor_scales # [0.5, 1, 1.5, 2]
        ratios = self.config.anchor_ratios # [1, 1.5, 2]

        lengths = np.outer(scales, ratios).ravel()
        lengths = np.sort(np.unique(lengths)) # [0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 3.0, 4.0]

        anchors = np.array([[width, height] for width in lengths for height in lengths])
        return anchors
    
    
    def decode_offsets(self, offsets, anchors, fm_shape):
        '''
        Transform the predicted offsets to absolute coordinates in the feature maps.
        
            Parameters:
                offsets (tf.Tensor): The predicted offsets for the anchors.
                anchors (np.array): The generated anchors for the RPN to predict offsets for.
                fm_shape (tf.Tensor): The shape of the feature maps.

            Returns:
                roi_boxes (tf.Tensor): The absolute coordinates in the feature maps.
        '''

        anchor_w = anchors[:, 0]
        anchor_h = anchors[:, 1]

        columns = []

        for x in range(fm_shape[1]):
            col_boxes = []

            for y in range(fm_shape[2]):
                dx = offsets[:, x, y, ::4]
                dy = offsets[:, x, y, 1::4]
                dw = offsets[:, x, y, 2::4]
                dh = offsets[:, x, y, 3::4]

                ctr_x, ctr_y = x + dx, y + dy
                w, h = anchor_w * tf.exp(dw), anchor_h * tf.exp(dh)

                x1, y1 = ctr_x - 0.5 * w, ctr_y - 0.5 * h
                x2, y2 = ctr_x + 0.5 * w, ctr_y + 0.5 * h

                boxes = tf.stack([x1, y1, x2, y2], axis=-1) # Shape: (batch_size, 64, 4)
                col_boxes.append(boxes)

            col_boxes = tf.concat(col_boxes, axis=1)  # Shape: (batch_size, 8*64, 4)
            columns.append(col_boxes)

        roi_boxes = tf.concat(columns, axis=1)  # Shape: (batch_size, 8*8*64, 4)            
        return roi_boxes
    
    
    def clip_boxes(self, boxes, fm_shape):
        '''
        Clip the predicted coordinates that are outside the feature maps.
        
            Parameters:
                boxes (tf.Tensor): The predicted absolute coordinates for the anchors.
                fm_shape (tf.Tensor): The shape of the feature maps.

            Returns:
                clipped_boxes (tf.Tensor): The clipped absolute coordinates in the feature maps.
        '''

        x1 = tf.clip_by_value(boxes[:, :, 0], 0, fm_shape[1])
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, fm_shape[2])
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, fm_shape[1])
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, fm_shape[2])

        clipped_boxes = tf.stack([x1, y1, x2, y2], axis=-1)
        return clipped_boxes
    
    
    def non_maximum_suppression(self, boxes, roi_scores):
        '''
        Apply Non-Maximum Suppression (NMS) to the proposed coordinates.
        
            Parameters:
                boxes (tf.Tensor): The predicted absolute coordinates for the anchors.
                roi_scores (tf.Tensor): The score whether an anchor contains an object or not.

            Returns:
                selected_boxes (tf.Tensor): The selected absolute coordinates in the feature maps.
        '''

        def single_image_nms(boxes, scores):
            selected_indices = tf.image.non_max_suppression(boxes, scores,
                                                            max_output_size=self.config.rpn_max_proposals,
                                                            iou_threshold=self.config.rpn_iou_threshold,
                                                            score_threshold=self.config.rpn_score_threshold)
            
            selected_boxes = tf.gather(boxes, selected_indices)
            return selected_boxes
        

        # NMS expects the boxes to be in the format (y1, x1, y2, x2)
        reshaped_boxes = tf.stack([boxes[:, :, 1], boxes[:, :, 0], boxes[:, :, 3], boxes[:, :, 2]], axis=-1)
        reshaped_scores = tf.reshape(roi_scores, (-1, 8*8*64)) # Shape: (batch_size, 8, 8, 64) -> (batch_size, 8*8*64)

        selected_boxes = []
        for i in range(reshaped_boxes.shape[0]):
            selected_image_boxes = single_image_nms(reshaped_boxes[i], reshaped_scores[i]) # Shape: (num_selected, 4)
            selected_boxes.append(selected_image_boxes)
        
        selected_boxes = tf.stack(selected_boxes, axis=0) # Shape: (batch_size, num_selected, 4)

        # Reformat the boxes to the format (x1, y1, x2, y2)
        selected_boxes = tf.stack([selected_boxes[:, :, 1], selected_boxes[:, :, 0], selected_boxes[:, :, 3], selected_boxes[:, :, 2]], axis=-1)
        return selected_boxes
