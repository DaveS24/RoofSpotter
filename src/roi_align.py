import tensorflow as tf
import time


class ROIAlignLayer:
    '''
    The ROI Align layer for the Mask R-CNN model.
    
        Attributes:
            config (Config): The configuration settings.
            backbone (Backbone): The backbone for the Mask R-CNN model.
            rpn (RPN): The Region Proposal Network (RPN) for the Mask R-CNN model.
            model (tf.keras.Model): The ROI Align layer.
            
        Methods:
            build_model: Link the components to build the ROI Align layer.
            place_sampling_points: Place the sampling points in the ROI boxes.
            bilinear_interpolate: Interpolate the feature map using the sampling points.
    '''

    def __init__(self, config, backbone, rpn, name='ROI_Align'):
        self.config = config
        self.backbone = backbone
        self.rpn = rpn

        self.model = self.build_model()
        self.model._name = name


    def build_model(self):
        '''
        Link the components to build the ROI Align layer.

            Parameters:
                None

            Returns:
                model (tf.keras.Model): The ROI Align layer.
        '''

        input_feature_map = tf.keras.layers.Input(shape=self.backbone.model.output.shape[1:], batch_size=self.config.batch_size,
                                                   name='input_feature_maps')
        input_roi_boxes = tf.keras.layers.Input(shape=self.rpn.model.output.shape[1:], batch_size=self.config.batch_size,
                                                name='input_roi_boxes')

        # Place the sampling points in the ROI boxes
        sampling_points = self.place_sampling_points(input_roi_boxes) # Shape: (batch_size, num_rois, sample_grid[0]*sample_grid[1], 2)

        # Interpolate the feature map using the sampling points
        interpolated_rois = self.bilinear_interpolate(input_feature_map, sampling_points) # Shape: (batch_size, num_rois, sample_grid[0], sample_grid[1], num_channels)

        # Pool the interpolations
        aligned_rois = self.pool_interpolations(interpolated_rois) # Shape: (batch_size, num_rois, pool_size, pool_size, num_channels)

        model = tf.keras.Model(inputs=[input_feature_map, input_roi_boxes], outputs=aligned_rois)
        return model
    

    def place_sampling_points(self, roi_boxes):
        '''
        Place the sampling points in the ROI boxes.

            Parameters:
                roi_boxes (tf.Tensor): The absolute coordinates of the Region of Interest (ROI) boxes.

            Returns:
                sampling_points (tf.Tensor): The sampling points in the ROI boxes.
        '''

        len_x = roi_boxes[:, :, 2] - roi_boxes[:, :, 0] # Shape: (batch_size, num_rois)
        len_y = roi_boxes[:, :, 3] - roi_boxes[:, :, 1]

        sample_grid = self.config.roi_align_sample_grid

        start_x = roi_boxes[:, :, 0] + len_x / (sample_grid[0] * 2) # Shape: (batch_size, num_rois)
        start_y = roi_boxes[:, :, 1] + len_y / (sample_grid[1] * 2)
        end_x = roi_boxes[:, :, 2] - len_x / (sample_grid[0] * 2) # Shape: (batch_size, num_rois)
        end_y = roi_boxes[:, :, 3] - len_y / (sample_grid[1] * 2)

        # Create the x and y coordinates for the sampling points
        sampling_x = tf.linspace(start_x, end_x, sample_grid[0], axis=-1) # Shape: (batch_size, num_rois, sample_grid[0])
        sampling_y = tf.linspace(start_y, end_y, sample_grid[1], axis=-1)
        shape_x = tf.shape(sampling_x)
        shape_y = tf.shape(sampling_y)

        # Broadcast the x and y coordinates to create the grid of sampling points
        sampling_x = tf.reshape(sampling_x, tf.concat([shape_x, [1, 1]], 0))                   # Shape: (batch_size, num_rois, sample_grid[0], 1, 1)
        sampling_y = tf.reshape(sampling_y, tf.concat([shape_y[:-1], [1, shape_y[-1], 1]], 0)) # Shape: (batch_size, num_rois, 1, sample_grid[1], 1)

        sampling_x = tf.broadcast_to(sampling_x, tf.concat([shape_x, [shape_y[-1], 1]], 0)) # Shape: (batch_size, num_rois, sample_grid[0], sample_grid[1], 1)
        sampling_y = tf.broadcast_to(sampling_y, tf.concat([shape_y, [shape_x[-1], 1]], 0)) # Shape: (batch_size, num_rois, sample_grid[1], sample_grid[0], 1)

        sampling_x = tf.reshape(sampling_x, tf.concat([shape_x[:-1], [shape_x[-1]*shape_y[-1], 1]], 0)) # Shape: (batch_size, num_rois, sample_grid[0]*sample_grid[1], 1)
        sampling_y = tf.reshape(sampling_y, tf.concat([shape_x[:-1], [shape_y[-1]*shape_x[-1], 1]], 0)) # Shape: (batch_size, num_rois, sample_grid[1]*sample_grid[0], 1)

        sampling_points = tf.concat([sampling_x, sampling_y], axis=-1) # Shape: (batch_size, num_rois, sample_grid[0]*sample_grid[1], 2)
        
        sampling_points += self.config.roi_align_sample_offset
        return sampling_points


    def bilinear_interpolate(self, feature_map, sampling_points):
        '''
        Interpolate the feature map using the sampling points.

            Parameters:
                feature_map (tf.Tensor): The feature map.
                sampling_points (tf.Tensor): The sampling points in the ROI boxes.

            Returns:
                interpolated_points (tf.Tensor): The interpolated points in the feature map.
        '''

        # Get the four corners in the feature map for the sampling points, Shape: (batch_size, num_rois, sample_grid[0]*sample_grid[1], 2)
        x_1 = tf.math.floor(sampling_points[..., 0])
        y_1 = tf.math.floor(sampling_points[..., 1])
        x_2 = tf.math.ceil(sampling_points[..., 0])
        y_2 = tf.math.ceil(sampling_points[..., 1])

        f_11 = tf.stack([x_1, y_1], axis=-1)
        f_12 = tf.stack([x_2, y_1], axis=-1)
        f_21 = tf.stack([x_1, y_2], axis=-1)
        f_22 = tf.stack([x_2, y_2], axis=-1)

        # Calculate the weights for the four corners, Shape: (batch_size, num_rois, sample_grid[0]*sample_grid[1])
        w_11 = (x_2 - sampling_points[..., 0]) * (y_2 - sampling_points[..., 1])
        w_12 = (sampling_points[..., 0] - x_1) * (y_2 - sampling_points[..., 1])
        w_21 = (x_2 - sampling_points[..., 0]) * (sampling_points[..., 1] - y_1)
        w_22 = (sampling_points[..., 0] - x_1) * (sampling_points[..., 1] - y_1)

        w_11 = tf.expand_dims(w_11, axis=-1)
        w_12 = tf.expand_dims(w_12, axis=-1)
        w_21 = tf.expand_dims(w_21, axis=-1)
        w_22 = tf.expand_dims(w_22, axis=-1)

        f_11 = tf.cast(f_11, tf.int32)
        f_12 = tf.cast(f_12, tf.int32)
        f_21 = tf.cast(f_21, tf.int32)
        f_22 = tf.cast(f_22, tf.int32)

        # Perform bilinear interpolation using the four corners and the corresponding weights, Shape: (batch_size, num_rois, sample_grid[0]*sample_grid[1], num_channels)
        interpolated_points = w_11 * tf.gather_nd(feature_map, f_11, batch_dims=1) + \
                              w_12 * tf.gather_nd(feature_map, f_12, batch_dims=1) + \
                              w_21 * tf.gather_nd(feature_map, f_21, batch_dims=1) + \
                              w_22 * tf.gather_nd(feature_map, f_22, batch_dims=1)
        
        # Reshape the interpolated points, Shape: (batch_size, num_rois, sample_grid[0], sample_grid[1], num_channels)
        sample_grid = self.config.roi_align_sample_grid
        interpolated_points = tf.reshape(interpolated_points, [feature_map.shape[0], -1, sample_grid[0], sample_grid[1], feature_map.shape[-1]])
        return interpolated_points
    

    def pool_interpolations(self, interpolated_rois):
        aligned_rois = interpolated_rois
        return aligned_rois
