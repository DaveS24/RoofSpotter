class Config:
    '''
    Configuration class for Mask R-CNN.
    
        Attributes:
            image_shape (tuple): The shape of the input image.
            num_classes (int): The number of classes.
            batch_size (int): The batch size.
            model_dir (str): The directory to save the model.
            anchor_scales (list): The scales for the anchors.
            anchor_ratios (list): The ratios for the anchors.
            backbone_weights (str): The weights for the backbone.
            backbone_trainable_layers (int): The number of trainable layers in the backbone.
            rpn_conv_filters (int): The number of filters for the RPN convolutional layer.
            rpn_max_proposals (int): The maximum number of proposals for the RPN.
            rpn_iou_threshold (float): The IoU threshold for the RPN.
            rpn_score_threshold (float): The score threshold for the RPN.
            rpn_optimizer (str): The optimizer for the RPN.
            roi_align_pool_size (tuple): The pool size for the ROI Align layer.
            classifier_dense_units (int): The number of units for the classifier dense layer.
            classifier_optimizer (str): The optimizer for the classifier.
            mask_head_conv_filters (int): The number of filters for the mask head convolutional layer.
            mask_head_upsample_filters (int): The number of filters for the mask head upsampling layer.
            mask_head_optimizer (str): The optimizer for the mask head.
            
        Methods:
            info: Print the configuration settings.
    '''

    def __init__(self):
        # General
        self.image_shape = (250, 250, 3)
        self.num_classes = 2
        self.batch_size = 32
        self.model_dir = '../model/'

        # Utils
        self.anchor_scales = [0.5, 1, 1.5, 2] # Due to ResNet50's downsampling by 250 / 8 = 31.25 -> [16, 32, 48, 64] px in the image space
        self.anchor_ratios = [1, 1.5, 2]

        # Backbone
        self.backbone_weights = 'imagenet'
        self.backbone_trainable_layers = 3

        # RPN
        self.rpn_conv_filters = 512
        self.rpn_max_proposals = 100
        self.rpn_iou_threshold = 0.7
        self.rpn_score_threshold = 0.5
        self.rpn_optimizer = 'adam'

        # ROI Align
        self.roi_align_pool_size = (7, 7)

        # Classifier
        self.classifier_dense_units = 1024
        self.classifier_optimizer = 'adam'

        # Mask Head
        self.mask_head_conv_filters = 256
        self.mask_head_upsample_filters = 256
        self.mask_head_optimizer = 'adam'
        

    def info(self):
        '''
        Print the configuration settings.
        
            Parameters:
                None
                
            Returns:
                None
        '''

        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
