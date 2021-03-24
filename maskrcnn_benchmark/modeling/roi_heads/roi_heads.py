# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .box_head.box_head_part import build_roi_box_head_part
from .box_head.box_head_padreg import build_roi_box_head_padreg
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, features, proposals, targets=None, query=False):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        
        if not self.training or not (self.cfg.MODEL.APNET.TRAIN_PART or self.cfg.MODEL.APNET.TRAIN_PADREG):
            x, detections, loss_box = self.box(features, proposals, targets, query)
            losses.update(loss_box)
        # if query
        # proposals is targets

        # when training
        # the proposals(detections) is the 128 filtered from RPN

        # when testing
        # the detections is the ultimate result (post_processed) 


        if self.cfg.MODEL.APNET.BOX_PADREG_ON:
            loss_box_pad_reg = {}
            if self.training and self.cfg.MODEL.APNET.TRAIN_PADREG:
                x, detections, loss_box_pad_reg = self.box_padreg(features, proposals, targets)
            elif not self.training:
                x, detections, loss_box_pad_reg = self.box_padreg(features, detections, targets, query)
            losses.update(loss_box_pad_reg)


        if self.cfg.MODEL.APNET.BOX_PART_ON:
            loss_box_part = {}
            if self.training and self.cfg.MODEL.APNET.TRAIN_PART:
                x, detections, loss_box_part = self.box_part(features, proposals, targets)
                # proposals = targets in box_head_part.py
                # train the head with gt bboxes
            elif not self.training:
                # if testing
                # for query, use targets
                # for gallery, use detections
                x, detections, loss_box_part = self.box_part(features, detections, targets, query)
            losses.update(loss_box_part)

        
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.APNET.BOX_PART_ON:
        roi_heads.append(("box_part", build_roi_box_head_part(cfg, in_channels)))
    if cfg.MODEL.APNET.BOX_PADREG_ON:
        roi_heads.append(["box_padreg", build_roi_box_head_padreg(cfg, in_channels)])


    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
