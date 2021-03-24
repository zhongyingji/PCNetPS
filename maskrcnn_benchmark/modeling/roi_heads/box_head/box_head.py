# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.layers import OIMLoss


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()

        self.oimloss = OIMLoss(cfg.MODEL.REID_CONFIG.FEAT_DIM, cfg.MODEL.REID_CONFIG.NUM_IDS)

        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg, self.oimloss)

    def forward(self, features, proposals, targets=None, query=False):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)


        x = self.feature_extractor(features, proposals)

        # final classifier that converts the features into predictions
        
        class_logits, box_regression, embed_norm = self.predictor(x)


        if not self.training:

            if query:
                # no filtering
                # proposals is target

                result = self.post_processor.forward_query((class_logits, box_regression, embed_norm), proposals)
                return x, result, {}

            else: 
                
                result = self.post_processor.forward_gallery((class_logits, box_regression, embed_norm), proposals)
                

                return x, result, {}

        loss_classifier, loss_box_reg, loss_oim_reid = self.loss_evaluator(
            [class_logits], [box_regression], 
            [embed_norm]
        )

        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, 
                loss_oim_reid=loss_oim_reid) ,
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)