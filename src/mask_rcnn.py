import tensorflow as tf

from backbone import Backbone
from rpn import RPN
from roi_align import ROIAlignLayer
from classifier import Classifier
from mask_head import MaskHead
from utils import Config


class MaskRCNN:
    def __init__(self, config=Config()):
        self.config = config

        # Initialize the components
        self.backbone = Backbone(self.config.input_shape, self.config.trainable_layers)
        print("Backbone model input shape:", self.backbone.model.input_shape)
        print("Backbone model output shape:", self.backbone.model.output_shape)

        self.rpn = RPN(self.backbone, self.config.input_shape)
        print("RPN model input shape:", self.rpn.model.input_shape)
        print("RPN model output shape:", self.rpn.model.output_shape)

        self.roi_align_layer = ROIAlignLayer(self.backbone, self.config.pool_size, self.config.num_rois)
        print("ROI Align layer input shape:", self.roi_align_layer.input_shape)
        print("ROI Align layer output shape:", self.roi_align_layer.output_shape)

        self.classifier = Classifier(self.roi_align_layer, self.config.num_classes)
        print("Classifier model input shape:", self.classifier.model.input_shape)
        print("Classifier model output shape:", self.classifier.model.output_shape)

        self.mask_head = MaskHead(self.roi_align_layer, self.config.num_classes)
        print("Mask head model input shape:", self.mask_head.model.input_shape)
        print("Mask head model output shape:", self.mask_head.model.output_shape)

        # Build the Mask R-CNN model
        self.model = self.build_model()

    def build_model(self):
        # Get the input image
        input_image = tf.keras.layers.Input(shape=self.config['input_shape'])

        # Get the feature map from the backbone
        feature_map = self.backbone(input_image)

        # Get the ROIs from the RPN
        rois = self.rpn(feature_map)

        # Get the ROI-aligned feature maps from the ROI Align layer
        roi_aligned = self.roi_align_layer([feature_map, rois])

        # Get the class scores and bounding box coordinates from the classifier
        class_scores, bbox = self.classifier(roi_aligned)

        # Get the binary masks from the mask head
        masks = self.mask_head(roi_aligned)

        # Create the model
        model = tf.keras.Model(inputs=input_image, outputs=[rois, class_scores, bbox, masks])
        return model
    
    def compile_model(self):
        self.rpn.compile_model(self.config['rpn_optimizer'])
        self.classifier.compile_model(self.config['classifier_optimizer'])
        self.mask_head.compile_model(self.config['mask_head_optimizer'])


# TODO:
    
# The Road to a Working Model

#     Complete the Components:  Carefully revisit our earlier discussions and fill in the missing pieces within each of your component files.

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
