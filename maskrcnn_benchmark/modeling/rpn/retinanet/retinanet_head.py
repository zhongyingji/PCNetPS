import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class RetinaNetHead(torch.nn.Module):

    def __init__(self, cfg, in_channels, cls_reg_shared=False):
        super(RetinaNetHead, self).__init__()

        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        self.num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        num_det_convs = cfg.MODEL.RETINANET.NUM_CONVS
        head_conv_dim = cfg.MODEL.RETINANET.HEAD_CONV_DIM

        num_offset_convs = cfg.MODEL.PCNET.NUM_OFFSET_CONVS
        offset_conv_dim = cfg.MODEL.RETINANET.HEAD_OFFSET_CONV_DIM

        num_points = cfg.MODEL.PCNET.NUM_POINTS
        num_group_points = cfg.MODEL.PCNET.NUM_GROUP_POINTS

       	det_conv_dim = [in_channels] + [head_conv_dim] * num_det_convs
       	offset_conv_dim = [in_channels] + [offset_conv_dim] * num_offset_convs

       	self.in_channels = in_channels

        cls_tower = []
        for i in range(num_det_convs):
            cls_tower.append(
                nn.Conv2d(
                    det_conv_dim[i],
                    det_conv_dim[i+1],
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))

        if not cls_reg_shared:
            bbox_tower = []
            for i in range(num_det_convs):
                bbox_tower.append(
                    nn.Conv2d(
                        det_conv_dim[i],
                        det_conv_dim[i+1],
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                bbox_tower.append(nn.ReLU())
            self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
            initializer = [self.bbox_tower]
        else:
            # if sharing cls and reg tower, rewrite the forward function
            self.bbox_tower = None
            initializer = []
            
        self.cls_logits = nn.Conv2d(
            det_conv_dim[-1], num_classes,
            kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            det_conv_dim[-1], 4, 
            kernel_size=3, stride=1, padding=1
        )

        offset_tower = []
        for i in range(num_offset_convs):
            offset_tower.append(
                nn.Conv2d(
                    offset_conv_dim[i],
                    offset_conv_dim[i+1],
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            offset_tower.append(nn.ReLU())

        self.add_module('offset_tower', nn.Sequential(*offset_tower))

        offset_out_dim = self.num_anchors * num_group_points * 2 * num_points
        self.offset_coord = nn.Conv2d(
        	offset_conv_dim[-1], offset_out_dim, # 2*9*3
            kernel_size=3, stride=1, padding=1)

        initializer = initializer + [self.cls_tower, self.cls_logits, self.bbox_pred, self.offset_tower]

        self.points_feat_dim = cfg.MODEL.REID_CONFIG.FEAT_DIM
        if self.points_feat_dim > in_channels:
            self.bridge_reid_det = nn.Sequential(
                nn.Conv2d(self.points_feat_dim, in_channels, kernel_size=3, stride=1, padding=1), 
                nn.ReLU()
            )
            initializer.append(self.bridge_reid_det)

        # Initialization
        for modules in initializer:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        for l in [self.offset_coord,]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.)

        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        from maskrcnn_benchmark.layers import Scale
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(cfg.MODEL.RPN.ANCHOR_STRIDE))])

    def forward(self, x, reid_feat_func, centers):
        bbox_offset, reid_feat, inter_feat = self.get(x, reid_feat_func, centers)

        logits = []
        bbox_reg = []
        for idx, feature in enumerate(inter_feat):
            n, _, h, w = feature.shape 
            regroup_feature = feature.view(n*self.num_anchors, -1, h, w) # (nxn_anchor, feat_dim, h, w)
            
            logit = self.cls_logits(self.cls_tower(regroup_feature))
            reg = self.bbox_pred(self.bbox_tower(regroup_feature))

            logits.append(logit.view(n, -1, h, w)) # (n, n_anchor, h, w)
            bbox_reg.append(reg.view(n, -1, h, w)) # (n, n_anchorx4, h, w)

        return logits, bbox_reg, bbox_offset, inter_feat, reid_feat

    def get(self, x, reid_feat_func, centers):
        bbox_offset = []
        for idx, feature in enumerate(x):
            bbox_offset.append(self.scales[idx](self.offset_coord(self.offset_tower(feature))))

        reid_feat = reid_feat_func(x, bbox_offset, centers)

        if self.points_feat_dim > self.in_channels:
            inter_feat = [self.bridge_reid_det(per_inter_feat) for per_inter_feat in reid_feat]
        else:
            inter_feat = reid_feat

        return bbox_offset, reid_feat, inter_feat



class RetinaNetHeadShareClsReg(RetinaNetHead):
    def __init__(self, cfg, in_channels):
        super(RetinaNetHeadShareClsReg, self).__init__(cfg, in_channels, cls_reg_shared=True)

    def forward(self, x, reid_feat_func, centers):
        bbox_offset, reid_feat, inter_feat = self.get(x, reid_feat_func, centers)

        logits = []
        bbox_reg = []
        for idx, feature in enumerate(inter_feat):
            n, _, h, w = feature.shape 
            regroup_feature = feature.view(n*self.num_anchors, -1, h, w) # (nxn_anchor, feat_dim, h, w)

            shared_cls_reg_feat = self.cls_tower(regroup_feature)

            logit = self.cls_logits(shared_cls_reg_feat)
            reg = self.bbox_pred(shared_cls_reg_feat)

            logits.append(logit.view(n, -1, h, w)) # (n, n_anchor, h, w)
            bbox_reg.append(reg.view(n, -1, h, w)) # (n, n_anchorx4, h, w)

        return logits, bbox_reg, bbox_offset, inter_feat, reid_feat


def build_retinanet_head(cfg, in_channels):
    if cfg.MODEL.RETINANET.HEAD_SHARE_CLS_REG:
        return RetinaNetHeadShareClsReg(cfg, in_channels)
    else:
        return RetinaNetHead(cfg, in_channels)