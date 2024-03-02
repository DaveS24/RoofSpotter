import tensorflow as tf

from utils import AnchorGenerator


class RPN:
    def __init__(self, backbone, input_shape=(250, 250, 3)):
        self.backbone = backbone
        self.input_shape = input_shape
        self.anchors = AnchorGenerator.generate_anchors()
        self.num_anchors = len(self.anchors)
        self.model = self.build_model()

    def apply_regressions(self, x_class, x_regr, anchors):
        # Calculate objectness scores
        objectness_scores = tf.nn.softmax(x_class)[:, 1]

        # Apply regression adjustments to the anchors
        dx = x_regr[:, 0] * anchors[:, 2]
        dy = x_regr[:, 1] * anchors[:, 3]
        dw = tf.exp(x_regr[:, 2]) * anchors[:, 2]
        dh = tf.exp(x_regr[:, 3]) * anchors[:, 3]

        # Compute the coordinates of the refined anchors
        refined_anchors = tf.stack([
            anchors[:, 0] + dx - dw / 2,
            anchors[:, 1] + dy - dh / 2,
            anchors[:, 0] + dx + dw / 2,
            anchors[:, 1] + dy + dh / 2
        ], axis=-1)

        return objectness_scores, refined_anchors
    
    def non_maximum_suppression(self, scores, boxes, max_output_size, iou_threshold=0.5, score_threshold=0.0):
        # Get the indices of the boxes to keep
        indices = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold, score_threshold)

        # Select the boxes and scores
        selected_boxes = tf.gather(boxes, indices)
        selected_scores = tf.gather(scores, indices)

        return selected_boxes, selected_scores

    def build_model(self):
        # Get the feature map from the backbone
        feature_map = self.backbone.model.output

        # Apply a 3x3 convolutional layer
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv')(feature_map)
        x = tf.keras.layers.BatchNormalization()(x)

        # Classification layer - 2 outputs per anchor (objectness score)
        x_class = tf.keras.layers.Conv2D(self.num_anchors * 2, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)

        # Regression layer - 4 outputs per anchor (bounding box coordinates)
        x_regr = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regr')(x)

        # Get the shape of the feature map
        shape = tf.shape(feature_map)

        # Compute the stride of the feature map
        stride = self.input_shape[0] // shape[0]

        # Generate a grid of points in the feature map coordinates
        grid_y, grid_x = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing='ij')

        # Convert the grid to image coordinates
        grid_y = stride * grid_y
        grid_x = stride * grid_x

        # Map the anchors to each point in the grid
        anchors = tf.stack([grid_y - self.anchors[:, 0] / 2,
                            grid_x - self.anchors[:, 1] / 2,
                            grid_y + self.anchors[:, 2] / 2,
                            grid_x + self.anchors[:, 3] / 2], axis=-1)

        # Reshape the anchors and the outputs to the same shape
        anchors = tf.reshape(anchors, [-1, 4])
        x_class = tf.reshape(x_class, [-1, 2])
        x_regr = tf.reshape(x_regr, [-1, 4])

        x_class, x_regr, anchors = self.model.output
        objectness_scores, refined_anchors = self.apply_regressions(x_class, x_regr, anchors)
        selected_boxes, selected_scores = self.non_maximum_suppression(objectness_scores, refined_anchors, max_output_size=300)
        return tf.keras.Model(inputs=self.model.input, outputs=[selected_scores, selected_boxes])
    
    def smooth_l1_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        mask = tf.cast(tf.less(abs_loss, 1.0), 'float32')
        return mask * sq_loss + (1-mask) * (abs_loss - 0.5)

    def rpn_loss(self, y_true_class, y_true_regr, y_pred_class, y_pred_regr):
        # Classification loss
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true_class, logits=y_pred_class))

        # Regression loss
        regr_loss = tf.reduce_mean(self.smooth_l1_loss(y_true_regr, y_pred_regr))

        # Total loss
        total_loss = class_loss + regr_loss

        return total_loss
    
    def compile_model(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=self.rpn_loss)
