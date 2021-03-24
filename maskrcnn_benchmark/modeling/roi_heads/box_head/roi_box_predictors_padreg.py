# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import torch.nn.functional as F


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor_padreg")
class FastRCNNPredictor_padreg(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor_padreg, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.reg = nn.Linear(num_inputs, 1)


    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #reg_val = F.sigmoid(self.reg(x))

        reg_val = F.tanh(self.reg(x).squeeze(1))
        return reg_val



@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor_padregBidirection")
class FastRCNNPredictor_padregBidirection(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor_padregBidirection, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.reg = nn.Linear(num_inputs, 2)


    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        reg_val = F.tanh(self.reg(x))

        return reg_val




def make_roi_box_predictor(cfg, in_channels):
    use_padreg_bidirection = cfg.MODEL.REID.USE_PADREG_BIDIRECTION 
    func = FastRCNNPredictor_padregBidirection if use_padreg_bidirection else FastRCNNPredictor_padreg
    # func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PADREG_PREDICTOR]
    return func(cfg, in_channels)











