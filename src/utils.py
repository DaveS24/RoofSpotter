import numpy as np


class Config:
    def __init__(self):
        self.image_shape = (250, 250, 3)
        self.trainable_layers = 3
        self.pool_size = (7, 7)
        self.num_classes = 2
        self.rpn_optimizer = 'adam'
        self.classifier_optimizer = 'adam'
        self.mask_head_optimizer = 'adam'


class AnchorGenerator:
    @classmethod
    def generate_anchors(cls, base_size=8, ratios=[1, 2], scales=[1, 2, 4, 8]):
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = cls._ratio_enum(base_anchor, ratios)
        anchors = np.vstack([cls._scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
        return anchors
    
    @classmethod
    def _ratio_enum(cls, anchor, ratios):
        w, h, x_ctr, y_ctr = cls._ctr_and_size(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = cls._make_anchors(ws, hs, x_ctr, y_ctr)
        return anchors

    @classmethod
    def _scale_enum(cls, anchor, scales):
        w, h, x_ctr, y_ctr = cls._ctr_and_size(anchor)
        ws = np.array(w) * scales
        hs = np.array(h) * scales
        anchors = cls._make_anchors(ws, hs, x_ctr, y_ctr)
        return anchors
    
    @staticmethod
    def _ctr_and_size(anchor):
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    @staticmethod
    def _make_anchors(ws, hs, x_ctr, y_ctr):
        anchors = np.vstack([x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)]).transpose()
        return anchors
