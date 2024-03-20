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

        input_feature_maps = tf.keras.layers.Input(shape=self.backbone.model.output.shape[1:], batch_size=self.config.batch_size,
                                                   name='input_feature_maps')
        input_roi_boxes = tf.keras.layers.Input(shape=self.rpn.model.output.shape[1:], batch_size=self.config.batch_size,
                                                name='input_roi_boxes')
        
        aligned_rois = ...

        model = tf.keras.Model(inputs=[input_feature_maps, input_roi_boxes], outputs=aligned_rois)
        return model
