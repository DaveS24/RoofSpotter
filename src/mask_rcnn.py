import numpy as np
import random
import tensorflow as tf

from backbone import Backbone
from rpn import RPN
from roi_align import ROIAlignLayer
from classifier import Classifier
from mask_head import MaskHead
from utils import Config

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


class MaskRCNN:
    '''
    The Mask R-CNN model.
    
        Attributes:
            config (Config): The configuration settings.
            backbone (Backbone): The backbone for the Mask R-CNN model.
            rpn (RPN): The Region Proposal Network (RPN) for the Mask R-CNN model.
            roi_align (ROIAlignLayer): The ROI Align layer for the Mask R-CNN model.
            classifier (Classifier): The classifier for the Mask R-CNN model.
            mask_head (MaskHead): The mask head for the Mask R-CNN model.
            model (tf.keras.Model): The Mask R-CNN model.
            
        Methods:
            build_model: Link the components to build the Mask R-CNN model.
    '''

    def __init__(self):
        self.config = Config()

        self.backbone = Backbone(self.config)
        self.rpn = RPN(self.config, self.backbone)
        # self.roi_align = ROIAlignLayer(self.config, self.backbone, self.rpn)
        # self.classifier = Classifier(self.config, self.roi_align)
        # self.mask_head = MaskHead(self.config, self.roi_align)

        self.model = self.build_model()


    def build_model(self):
        '''
        Link the components to build the Mask R-CNN model.
        
            Parameters:
                None
                
            Returns:
                model (tf.keras.Model): The Mask R-CNN model.
        '''

        # Get the input image
        input_image = tf.keras.layers.Input(shape=self.config.image_shape, batch_size=self.config.batch_size, name='input_image')

        # Get the feature map from the backbone
        feature_map = self.backbone.model(input_image)

        # Get the ROIs from the RPN
        roi_boxes = self.rpn.model(feature_map)

        model = tf.keras.Model(inputs=input_image, outputs=roi_boxes, name='Mask_RCNN')

        # # Get the aligned ROIs from the ROI Align layer
        # aligned_rois = self.roi_align.model([feature_maps, roi_boxes])

        # # Get the class scores and bounding box coordinates from the classifier
        # class_scores, bbox = self.classifier.model(aligned_rois)

        # # Get the binary masks from the mask head
        # binary_masks = self.mask_head.model(aligned_rois)

        # model = tf.keras.Model(inputs=input_image, outputs=[class_scores, bbox, binary_masks], name='Mask_RCNN')
        return model

# TODO:
    
# The Road to a Working Model

#     Complete the Components

#     Training Configuration:
#         Decide on your dataset loading, data splitting, and augmentation procedures.
#         Choose an optimizer (e.g., Adam or SGD with momentum).
#         Set a learning rate and consider a learning rate schedule.
#         Implement the Mask R-CNN's overall loss function, which will likely be a combination of the individual RPN, classifier, and mask losses.

#     Training: Write the training loop in TensorFlow. This involves:
#         Fetching batches of data.
#         Passing data through your Mask R-CNN model.
#         Calculating losses.
#         Updating model weights using backpropagation and your chosen optimizer.

#     Evaluation: Add code to evaluate your model's performance on your validation set using metrics like average precision (AP), which is standard for object detection and segmentation tasks.
