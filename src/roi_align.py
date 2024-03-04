import tensorflow as tf


class ROIAlignLayer:
    def __init__(self, backbone, pool_size=(7, 7), num_rois=32):
        self.backbone = backbone
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.layer = self.build_layer()

    def roi_align(self, inputs):
        # Separate the inputs
        feature_map = inputs[0]
        rois = inputs[1]
        rpn_proposals = inputs[2]

        # Ensure the RPN proposals are in the correct format [y1, x1, y2, x2]
        rpn_proposals = tf.stack([
            rpn_proposals[:, 1],  # y1
            rpn_proposals[:, 0],  # x1
            rpn_proposals[:, 3],  # y2
            rpn_proposals[:, 2]   # x2
        ], axis=-1)

        # Limit the number of RPN proposals to the same number as the ROIs
        rpn_proposals = tf.slice(rpn_proposals, [0, 0, 0], [-1, self.num_rois, -1])

        # Combine the ROIs and RPN proposals
        combined_rois = tf.concat([rois, rpn_proposals], axis=1)

        # Normalize the coordinates of the ROIs by the spatial dimension of the feature map
        spatial_dims = tf.cast(tf.shape(feature_map)[1:3], tf.float32)
        rois_norm = tf.stack([
            combined_rois[..., 0] / spatial_dims[0],  # y1
            combined_rois[..., 1] / spatial_dims[1],  # x1
            combined_rois[..., 2] / spatial_dims[0],  # y2
            combined_rois[..., 3] / spatial_dims[1]   # x2
        ], axis=-1)

        # Reshape the rois_norm tensor to be 2-D
        rois_norm = tf.reshape(rois_norm, [-1, 4])

        # Adjust the box_ind argument to match the new shape of rois_norm
        box_ind = tf.range(tf.shape(feature_map)[0])
        box_ind = tf.repeat(box_ind, tf.shape(rois_norm)[0] // tf.shape(feature_map)[0])

        # Use crop_and_resize to extract the ROIs and reduce them to the pool size
        rois_aligned = tf.image.crop_and_resize(feature_map, rois_norm, box_ind, self.pool_size)

        # Reshape the rois_aligned tensor back to its original shape
        rois_aligned = tf.reshape(rois_aligned, [-1, self.num_rois, self.pool_size[0], self.pool_size[1], tf.shape(feature_map)[-1]])

        return rois_aligned

    def build_layer(self):
        # Define the input layers
        feature_map_input = self.backbone.model.output
        rois_input = tf.keras.layers.Input(shape=(self.num_rois, 4))
        rpn_proposals_input = tf.keras.layers.Input(shape=(None, 4))

        # Apply the ROIAlign layer
        x = tf.keras.layers.Lambda(self.roi_align)([feature_map_input, rois_input, rpn_proposals_input])

        # Create the layer
        layer = tf.keras.Model(inputs=[feature_map_input, rois_input, rpn_proposals_input], outputs=x)
        return layer


# TODO:
    
# Key Considerations and Areas for Refinement

#     Bilinear Interpolation:
#         Crucially, verify that the crop_and_resize operation is configured to use bilinear interpolation by setting the method argument to 'bilinear'. This is essential for the correct functioning of ROI Align.

#     Compatibility:
#         Double-check the output shape of your ROI Align layer. Ensure that the dimensions of the extracted ROI features match the expectations of your subsequent Mask Head component.

# Recommendations

#     Testing: Similar to the RPN, I'd recommend temporarily connecting your ROI Align to your other components (backbone, RPN). Pass sample images and proposed regions to validate that data flows correctly, and the output of ROI Align has the expected shape.

#     Original ROIs: Consider if there's a need to separate the aligned features generated from the original ROIs and those that came from the RPN later in the pipeline. You might add logic to keep track of this if necessary.
