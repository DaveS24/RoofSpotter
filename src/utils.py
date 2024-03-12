import numpy as np
import tensorflow as tf


class Config:
    def __init__(self):
        # General
        self.image_shape = (250, 250, 3)
        self.num_classes = 2
        self.batch_size = 32
        self.model_dir = '../model/'

        # Utils
        self.anchor_base_size = 8
        self.anchor_ratios = [0.5, 0.75, 1, 2]
        self.anchor_scales = [1, 2, 4, 8]

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
