import tensorflow as tf

from backbone import Backbone
from rpn import RPN
from roi_align import ROIAlignLayer
from classifier import Classifier
from mask_head import MaskHead
from utils import Config


class MaskRCNN:
    def __init__(self):
        self.config = Config()

        # Initialize the components
        self.backbone = Backbone(self.config)
        self.rpn = RPN(self.config, self.backbone)
        self.roi_align_layer = ROIAlignLayer(self.config, self.backbone, self.rpn)
        self.classifier = Classifier(self.config, self.roi_align_layer)
        self.mask_head = MaskHead(self.config, self.roi_align_layer)

        self.model = self.build_model()

    def build_model(self):
        # Get the input image
        input_image = tf.keras.layers.Input(shape=self.config.image_shape, name='input_image')

        # Get the feature maps from the backbone
        feature_maps = self.backbone.model(input_image)

        # Get the ROIs from the RPN
        roi_boxes = self.rpn.model(feature_maps)

        # Get the aligned ROIs from the ROI Align layer
        aligned_rois = self.roi_align_layer.model([feature_maps, roi_boxes])

        # Get the class scores and bounding box coordinates from the classifier
        class_scores, bbox = self.classifier.layer(aligned_rois)

        # Get the binary masks from the mask head
        binary_masks = self.mask_head.model(aligned_rois)

        model = tf.keras.Model(inputs=input_image, outputs=[roi_boxes, aligned_rois, class_scores, bbox, binary_masks], name='Mask_RCNN')
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
