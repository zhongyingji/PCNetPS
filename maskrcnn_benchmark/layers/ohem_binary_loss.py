import torch
from torch import nn
import torch.nn.functional as F



class OHEMBinaryLoss(nn.Module):
    def __init__(self, batchsize, ratio):
        super(OHEMBinaryLoss, self).__init__()
        self.batchsize = batchsize
        self.ratio = ratio # positive ratio

    def forward(self, logits, targets):
        idx_t = targets >= 0
        logits_t = logits.squeeze(-1)[idx_t]
        targets_t = targets[idx_t]
        nt = logits_t.size(0)

        # cls_loss = F.cross_entropy(logits_t, targets_t, reduction='none')
        cls_loss = \
            F.binary_cross_entropy(F.sigmoid(logits_t), targets_t.to(torch.float), reduction='none')


        positive = torch.nonzero(targets_t >= 1).squeeze(1)
        negative = torch.nonzero(targets_t == 0).squeeze(1)

        num_pos = int(self.batchsize * self.ratio)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batchsize - num_pos

        sampled_pos_loss, _ = torch.sort(cls_loss[positive], descending=True)
        sampled_neg_loss, _ = torch.sort(cls_loss[negative], descending=True)
        sampled_pos_loss = sampled_pos_loss[:num_pos]
        sampled_neg_loss = sampled_neg_loss[:num_neg]

        return (sampled_pos_loss.sum() + sampled_neg_loss.sum()) / self.batchsize
