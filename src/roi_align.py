import tensorflow as tf


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
        
        # Get the feature maps from the backbone
        feature_maps = self.backbone.model.output
        # Get the ROI boxes from the RPN
        roi_boxes = self.rpn.model.output

        # Normalize the coordinates of the rois
        normalized_rois = roi_boxes / tf.constant([self.backbone.model.input_shape[1],
                                                   self.backbone.model.input_shape[2],
                                                   self.backbone.model.input_shape[1],
                                                   self.backbone.model.input_shape[2]], dtype=tf.float32)

        # Use crop_and_resize to perform the ROI Align operation
        aligned_rois = tf.image.crop_and_resize(feature_maps, normalized_rois, tf.range(tf.shape(roi_boxes)[0]), crop_size=self.config.roi_align_pool_size)

        model = tf.keras.Model(inputs=[feature_maps, roi_boxes], outputs=aligned_rois)
        return model
