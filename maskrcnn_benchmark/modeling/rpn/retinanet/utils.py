# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Utility functions minipulating the prediction layers
"""

from ..utils import cat

import torch

def l2norm(x):
    return x.div(x.norm(p=2, dim=1, keepdim=True).expand_as(x)+1e-8)


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression, box_offset, box_dconv_feat):
    box_cls_flattened = []
    box_regression_flattened = []
    box_offset_flattened = []
    box_dconv_feat_flattened = []

    
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression

    for box_cls_per_level, box_regression_per_level, box_offset_per_level, box_dconv_feat_per_level in zip(
        box_cls, box_regression, box_offset, box_dconv_feat
    ):  
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level) # (N, HWA, C)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)

        AxFEAT_DIM = box_dconv_feat_per_level.size(1)
        FEAT_DIM = AxFEAT_DIM // A
        box_dconv_feat_per_level = permute_and_flatten(
            box_dconv_feat_per_level, N, A, FEAT_DIM, H, W
        )
        box_dconv_feat_flattened.append(box_dconv_feat_per_level)

        AxOFFSET_DIM = box_offset_per_level.size(1)
        OFFSET_DIM = AxOFFSET_DIM // A
        box_offset_per_level = permute_and_flatten(
            box_offset_per_level, N, A, OFFSET_DIM, H, W
        )
        box_offset_flattened.append(box_offset_per_level)


    # [(N, H1W1A, C), (N, H2W2A, C), ...]

    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    box_dconv_feat_N = cat(box_dconv_feat_flattened, dim=1)
    box_dconv_feat = box_dconv_feat_N.reshape(-1, FEAT_DIM)
    box_offset = cat(box_offset_flattened, dim=1).reshape(-1, OFFSET_DIM)


    return box_cls, box_regression, box_offset, box_dconv_feat, box_dconv_feat_N

