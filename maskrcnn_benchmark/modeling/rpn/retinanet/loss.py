"""
This file contains specific functions for computing losses on the RetinaNet
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers
from .triplet import onlineTriplet
from .centerdist import Centerdist

from maskrcnn_benchmark.layers import smooth_l1_loss, SigmoidFocalLoss, OHEMBinaryLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rpn.loss import RPNLossComputation


class RetinaNetLossComputation(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 box_cls_loss_func,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0, 
                 add_gt=False,
                 triplet_margin=0.4
        ):
        
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = box_cls_loss_func
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels', 'ids']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

        self.add_gt = add_gt

        self.triplet_loss = onlineTriplet(triplet_margin)
        # self.center_loss = Centerdist()

    def add_gt_proposals(self, anchors, box_ext, box_adjs, targets):
        """
        This function is not used in RetinaNet
        """
        pass


    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        regression_targets_van = []
        ids = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image, ids_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            regression_targets_van_per_image = matched_targets.bbox

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            regression_targets_van.append(regression_targets_van_per_image)
            ids.append(ids_per_image)


        return labels, regression_targets, regression_targets_van, ids



    def __call__(self, anchors, centers, box_cls, box_regression, 
        box_offset, features_dconv, 
        reid_pool, targets, iteration=0, reid_feat=None, reid_bn=None):

        reid_loss_func = reid_pool["reid_loss_func"]
        reid_feat_func = reid_pool["reid_feat_func"]
        reid_feat_func_pooling = reid_pool["reid_feat_func_pooling"]

        if reid_feat is not None:
            box_dconv_feat = reid_feat
        else:
            box_dconv_feat = reid_feat_func(features_dconv, box_offset, centers)

        anchors_cat = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]

        labels, regression_targets, regression_targets_van, ids = self.prepare_targets(
                                                            anchors_cat, targets)
        # labels, [(H1W1A+H2W2A+... , ), ...]
        #               \  all images /

        N = len(labels)
        box_cls_flat, box_regression_flat, box_offset_flat, box_dconv_feat_flat, box_dconv_feat_flat_N = \
                concat_box_prediction_layers(box_cls, box_regression, box_offset, box_dconv_feat)
        # box_dconv_feat_flat_N: (N, H1W1A+H2W2A+..., dim)
        # (N, H1W1A+H2W2A+..., dim).view(-1, dim)

        split_pos_inds = [label > 0 for label in labels]
        regression_targets_van_pos = [
            regression_target_van[idx, :]
            for regression_target_van, idx in zip(regression_targets_van, split_pos_inds)
        ]

        box_dconv_feat_pos = [
            per_box_dconv_feat[idx, :]
            for per_box_dconv_feat, idx in zip(box_dconv_feat_flat_N, split_pos_inds)
        ]

        attentioned_feat_iou = reid_feat_func_pooling(
            features_dconv, 
            regression_targets_van_pos,
            box_dconv_feat_pos, 
            centers, 
            box_dconv_feat_flat_N
        )

        labels = torch.cat(labels, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        ids = torch.cat(ids, dim=0)
        pos_ids = ids[pos_inds].long()
        bn_attentioned_feat_iou = reid_bn[0](attentioned_feat_iou)
        l2_at_feat = self._l2norm(bn_attentioned_feat_iou) # (n_pos, feat_dim)
        retinanet_reid_loss, _ = reid_loss_func(
            l2_at_feat,                 
            pos_ids.clone().detach(),
            iteration
        )

        van_feat = box_dconv_feat_flat[pos_inds] # (n_pos, feat_dim)
        bn_van_feat = reid_bn[1](van_feat)
        l2_van_feat = self._l2norm(bn_van_feat)
        
        loss_triplet = self.triplet_loss(l2_at_feat, l2_van_feat, pos_ids.clone().detach())

        loss_center = None
        # loss_center = self.center_loss(l2_at_feat, pos_ids)

        regression_targets = torch.cat(regression_targets, dim=0)
        retinanet_regression_loss = smooth_l1_loss(
            box_regression_flat[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / (max(1, pos_inds.numel() * self.regress_norm))

        labels = labels.int()
        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls_flat,
            labels
        )
        if not hasattr(self.box_cls_loss_func, 'ratio'):
            retinanet_cls_loss /= (pos_inds.numel() + N)

        centers = [torch.cat(centers_per_image, dim=0) for centers_per_image in centers] # [(H1W1A+H2W2A+..., 3), ...]
        centers = torch.cat(centers, dim=0)
        regression_targets_van = torch.cat(regression_targets_van, dim=0)

        """
        points_dist_loss = reid_feat_func.get_points_dist_loss(
            centers[pos_inds], 
            box_offset_flat[pos_inds], # (n, 2*n_points)
            regression_targets_van[pos_inds]
        )
        """

        points_dist_loss = None

        return retinanet_cls_loss, retinanet_regression_loss, \
        retinanet_reid_loss, loss_triplet, \
        loss_center, points_dist_loss


    def _l2norm(self, x):
        return x.div(x.norm(p=2, dim=1, keepdim=True).expand_as(x)+1e-8)


def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    ids_per_image = matched_targets.get_field("ids")
    return labels_per_image, ids_per_image


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
        cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    box_cls_loss_func = SigmoidFocalLoss(
        cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA
    ) if not cfg.MODEL.RETINANET.OHEM_BINARY else OHEMBinaryLoss(
        cfg.MODEL.RETINANET.OHEM_BATCH_SIZE, 
        cfg.MODEL.RETINANET.OHEM_POS_FRACTION
    )

    loss_evaluator = RetinaNetLossComputation(
        matcher,
        box_coder,
        generate_retinanet_labels,
        box_cls_loss_func=box_cls_loss_func,
        bbox_reg_beta=cfg.MODEL.RETINANET.BBOX_REG_BETA,
        regress_norm=cfg.MODEL.RETINANET.BBOX_REG_WEIGHT,
        add_gt=cfg.MODEL.RETINANET.TRAIN_ADD_GT,
        triplet_margin=cfg.MODEL.REID_LOSS.TRIPLET_MARGIN
    )

    return loss_evaluator
