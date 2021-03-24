import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .retinanet_head import build_retinanet_head
from .inference import  make_retinanet_postprocessor
from .loss import make_retinanet_loss_evaluator

from .points_aggregation import make_points_aggregation_module
from .cross_scale_aggregation import make_cross_scale_aggregation_module

from ..anchor_generator import make_anchor_generator_retinanet
from ..point_generator import make_point_generator_retinanet

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.layers import OIMLoss


class RetinaNetModule(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator_retinanet(cfg)
        point_generator = make_point_generator_retinanet(cfg)

        # head selection
        head = build_retinanet_head(cfg, in_channels)

        box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)
        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.point_generator = point_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

        self.reid_loss_func = OIMLoss(
            cfg.MODEL.REID_CONFIG.FEAT_DIM, 
            cfg.MODEL.REID_CONFIG.NUM_IDS, 
            cq_size=cfg.MODEL.REID_LOSS.CQ_SIZE, 
            scalar=cfg.MODEL.REID_LOSS.OIM_SCALAR
        )

        self.reid_bn = nn.BatchNorm1d(cfg.MODEL.REID_CONFIG.FEAT_DIM)
        self.reid_bn_side = nn.BatchNorm1d(cfg.MODEL.REID_CONFIG.FEAT_DIM)

        # Contextual Points Pooling
        self.reid_feat_func = make_points_aggregation_module(cfg, in_channels)

        # Cross Scale Aggregation
        self.reid_feat_func_pooling = make_cross_scale_aggregation_module(cfg, in_channels)


        self.reid_pool = {"reid_loss_func": self.reid_loss_func, 
                            "reid_feat_func": self.reid_feat_func, 
                            "reid_feat_func_pooling": self.reid_feat_func_pooling, 
                            }

        self.w_cls = cfg.MODEL.RETINANET.CLS_WEIGHT
        self.w_reg = cfg.MODEL.RETINANET.REG_WEIGHT
        self.w_reid = cfg.MODEL.RETINANET.REID_WEIGHT
        self.w_tri = cfg.MODEL.RETINANET.TRI_WEIGHT
        self.w_ctr = cfg.MODEL.RETINANET.CENTER_WEIGHT


    def forward(self, images, features, targets=None, query=False, iteration=0):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        # assert isinstance(features_, list)
        # assert len(features_) == self.addition_fpn+1

        # features = features_[0]
        # features_dconv = features_[-1]

        anchors = self.anchor_generator(images, features)

        # for centers
        # [[f1*w1*3, f2*w2*3, ...], ...]
        #   \                    /
        #     all centers of one image
        #                   \        /
        #                    all images  
        centers = self.point_generator(images, features)

        box_cls, box_regression, box_offset, inter_feat, reid_feat = self.head(features, self.reid_feat_func, centers)
        # box_offset,
        # [(N, C, H1, W1), (N, C, H2, W2), ...]

        if self.training:
            return self._forward_train(anchors, centers, box_cls, box_regression, 
                box_offset, reid_feat, 
                self.reid_pool, targets, iteration, reid_feat=reid_feat, reid_bn=[self.reid_bn, self.reid_bn_side])
        else:
            return self._forward_test(anchors, centers, box_cls, box_regression, box_offset, 
                reid_feat,  
                self.reid_pool, targets, query, reid_feat=reid_feat, reid_bn=self.reid_bn)

    def _forward_train(self, anchors, centers, box_cls, box_regression, 
        box_offset, features, 
        reid_pool, targets, iteration=0, reid_feat=None, reid_bn=None):

        loss_box_cls, loss_box_reg, loss_reid, loss_triplet, loss_center, loss_points_dist = \
            self.loss_evaluator(
                anchors, centers, box_cls, box_regression, 
                box_offset, features,  
                reid_pool, targets, iteration, reid_feat=reid_feat, reid_bn=reid_bn
            )

        losses = {
            "loss_retina_cls": self.w_cls * loss_box_cls,
            "loss_retina_reg": self.w_reg * loss_box_reg,
            "loss_retina_reid": self.w_reid * loss_reid, 
        }

        if loss_points_dist is not None:
            losses["loss_points_dist"] = loss_points_dist
        if loss_triplet is not None:
            losses["loss_triplet"] = self.w_tri * loss_triplet
        if loss_center is not None:
            losses["loss_center"] = self.w_ctr * loss_center
    
        return anchors, losses

    def _forward_test(self, anchors, centers, box_cls, box_regression, 
        box_offset, features, reid_pool, targets, query, reid_feat=None, reid_bn=None):
        boxes = self.box_selector_test(anchors, centers, box_cls, box_regression, box_offset, features, 
            reid_pool, targets, query, reid_feat=reid_feat, reid_bn=reid_bn)
        return boxes, {}


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)
