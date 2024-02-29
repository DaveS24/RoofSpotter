import tensorflow as tf

from backbone import Backbone
from rpn import RPN
from roi_align import ROIAlign
from classifier import Classifier
from mask_head import MaskHead


class MaskRCNN:
    def __init__(self, config):
        self.config = config
        self.backbone = Backbone(self.config['input_shape'], self.config['trainable_layers'])
        self.rpn = RPN(self.backbone)
        self.roi_align = ROIAlign(self.backbone, self.config['pool_size'], self.config['num_rois'])
        self.classifier = Classifier(self.roi_align, self.config['num_classes'])
        self.mask_head = MaskHead(self.roi_align, self.config['num_classes'])
        self.model = self.build_model()

    def build_model(self):
        # Get the input image
        input_image = tf.keras.layers.Input(shape=self.config['input_shape'])

        # Get the feature map from the backbone
        feature_map = self.backbone(input_image)

        # Get the ROIs from the RPN
        rois = self.rpn(feature_map)

        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align([feature_map, rois])

        # Get the class scores and bounding box coordinates from the classifier
        class_scores, bbox = self.classifier(roi_aligned)

        # Get the binary masks from the mask head
        masks = self.mask_head(roi_aligned)

        # Create the model
        model = tf.keras.Model(inputs=input_image, outputs=[rois, class_scores, bbox, masks])
        return model
