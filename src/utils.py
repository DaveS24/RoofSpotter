import numpy as np


class Config:
    def __init__(self):
        self.input_shape = (250, 250, 3)
        self.trainable_layers = 3
        self.pool_size = (7, 7)
        self.num_rois = 32
        self.num_classes = 2
        self.rpn_optimizer = 'adam'
        self.classifier_optimizer = 'adam'
        self.mask_head_optimizer = 'adam'


class AnchorGenerator:
    @classmethod
    def generate_anchors(cls, base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = cls._ratio_enum(base_anchor, ratios)
        anchors = np.vstack([cls._scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
        return anchors

    @staticmethod
    def _ratio_enum(anchor, ratios):
        width, height, x_ctr, y_ctr = AnchorGenerator._ctr_and_size(anchor)
        size = width * height
        size_ratios = size / ratios

        widths = np.round(np.sqrt(size_ratios))
        heights = np.round(widths * ratios)
        anchors = AnchorGenerator._make_anchors(widths, heights, x_ctr, y_ctr)
        return anchors

    @staticmethod
    def _scale_enum(anchor, scales):
        width, height, x_ctr, y_ctr = AnchorGenerator._ctr_and_size(anchor)
        widths = width * scales
        heights = height * scales
        anchors = AnchorGenerator._make_anchors(widths, heights, x_ctr, y_ctr)
        return anchors

    @staticmethod
    def _ctr_and_size(anchor):
        width = anchor[2] - anchor[0] + 1
        height = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (width - 1)
        y_ctr = anchor[1] + 0.5 * (height - 1)
        return width, height, x_ctr, y_ctr

    @staticmethod
    def _make_anchors(widths, heights, x_ctr, y_ctr):
        widths = widths[:, np.newaxis]
        heights = heights[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (widths - 1),
                             y_ctr - 0.5 * (heights - 1),
                             x_ctr + 0.5 * (widths - 1),
                             y_ctr + 0.5 * (heights - 1)))
        return anchors
    