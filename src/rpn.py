import tensorflow as tf

from utils import AnchorGenerator


class RPN:
    def __init__(self, config, backbone, name='RPN'):
        self.config = config
        self.backbone = backbone
        
        self.anchors = AnchorGenerator.generate_anchors()
        self.num_anchors = len(self.anchors)
        self.model = self.build_model()
        self.model._name = name

    def build_model(self):
        # Get the feature map from the backbone
        feature_map = self.backbone.model.output
        shape = tf.shape(feature_map)

        # Apply a 3x3 convolutional layer
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv')(feature_map)
        x = tf.keras.layers.BatchNormalization()(x)

        # Classification output - 2 per anchor (objectness score)
        x_class = tf.keras.layers.Conv2D(self.num_anchors * 2, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)

        # Regression output - 4 per anchor (bounding box coordinates)
        x_regr = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regr')(x)

        # Generate a grid of points in the feature map coordinates
        grid_y, grid_x = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing='ij')

        # Convert the grid to image coordinates and map the anchors to each point in the grid
        anchors = self._compute_anchors(grid_y, grid_x, anchors=self.anchors, stride=shape[1])

        # Reshape the anchors and the outputs to the same shape
        anchors, x_class, x_regr = self._reshape_outputs(anchors, x_class, x_regr)

        objectness_scores, refined_anchors = self._apply_regressions(x_class, x_regr, anchors)
        selected_boxes, selected_scores = self._non_maximum_suppression(objectness_scores, refined_anchors, max_output_size=300)
        return tf.keras.Model(inputs=feature_map, outputs=[selected_scores, selected_boxes])
    
    @staticmethod
    def _compute_anchors(grid_y, grid_x, anchors, stride):
        # Convert the grid to image coordinates
        grid_y = grid_y * stride
        grid_x = grid_x * stride

        # Map the anchors to each point in the grid
        # Each anchor is defined by its top-left and bottom-right corners
        anchors = tf.stack([grid_y - anchors[:, 0] / 2,
                            grid_x - anchors[:, 1] / 2,
                            grid_y + anchors[:, 2] / 2,
                            grid_x + anchors[:, 3] / 2], axis=-1)
        return anchors
    
    @staticmethod
    def _reshape_outputs(anchors, x_class, x_regr):
        return tf.reshape(anchors, [-1, 4]), tf.reshape(x_class, [-1, 2]), tf.reshape(x_regr, [-1, 4])
    
    @staticmethod
    def _apply_regressions(x_class, x_regr, anchors):
        # Calculate objectness scores
        objectness_scores = tf.nn.softmax(x_class)[:, 1]

        anchors = tf.cast(anchors, tf.float32)

        # Apply regression adjustments to the anchors
        dx = x_regr[:, 0] * anchors[:, 2]
        dy = x_regr[:, 1] * anchors[:, 3]
        dw = tf.exp(x_regr[:, 2]) * anchors[:, 2]
        dh = tf.exp(x_regr[:, 3]) * anchors[:, 3]

        # Compute the coordinates of the refined anchors
        # Each anchor is defined by its top-left and bottom-right corners
        refined_anchors = tf.stack([
            anchors[:, 0] + dx - dw / 2,
            anchors[:, 1] + dy - dh / 2,
            anchors[:, 0] + dx + dw / 2,
            anchors[:, 1] + dy + dh / 2
        ], axis=-1)

        return objectness_scores, refined_anchors
    
    @staticmethod
    def _non_maximum_suppression(scores, boxes, max_output_size, iou_threshold=0.5, score_threshold=0.0):
        # Get the indices of the boxes to keep
        indices = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold, score_threshold)

        # Select the boxes and scores
        selected_boxes = tf.gather(boxes, indices)
        selected_scores = tf.gather(scores, indices)

        return selected_boxes, selected_scores
    
    def compile_model(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=self.rpn_loss)

    @classmethod
    def rpn_loss(cls, y_true_class, y_true_regr, y_pred_class, y_pred_regr):
        # Classification loss
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true_class, logits=y_pred_class))

        # Regression loss
        regr_loss = tf.reduce_mean(cls._smooth_l1_loss(y_true_regr, y_pred_regr))

        # Total loss
        total_loss = class_loss + regr_loss

        return total_loss
    
    @staticmethod
    def _smooth_l1_loss(y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        mask = tf.cast(tf.less(abs_loss, 1.0), 'float32')
        return mask * sq_loss + (1-mask) * (abs_loss - 0.5)    
