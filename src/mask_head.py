import tensorflow as tf


class MaskHead:
    '''
    The mask head for the Mask R-CNN model.
    
        Attributes:
            config (Config): The configuration settings.
            roi_align (ROIAlignLayer): The ROI Align layer for the Mask R-CNN model.
            model (tf.keras.Model): The mask head for the Mask R-CNN model.
            
        Methods:
            build_model: Link the components to build the mask head.
    '''

    def __init__(self, config, roi_align, name='Mask_Head'):
        self.config = config
        self.roi_align = roi_align
        
        self.model = self.build_model()
        self.model._name = name


    def build_model(self):
        '''
        Link the components to build the mask head.
        
            Parameters:
                None
                
            Returns:
                model (tf.keras.Model): The mask head for the Mask R-CNN model.
        '''

        input_roi_aligned = tf.keras.layers.Input(shape=self.roi_align.model.output.shape[1:], batch_size=self.config.batch_size,
                                                  name='input_roi_aligned')
        
        binary_masks = ...

        model = tf.keras.Model(inputs=input_roi_aligned, outputs=binary_masks)
        return model
    