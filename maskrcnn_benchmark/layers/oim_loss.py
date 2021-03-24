import torch
import torch.nn.functional as F
from torch import nn, autograd


class SoftmaxLoss(nn.Module):
    def __init__(self, num_features, num_labeled_pids, cq_size=5000, scalar=10.0, momentum=0.5,
                 weight=None, size_average=True):
        super(SoftmaxLoss, self).__init__()
        self.num_features = num_features
        self.num_labeled_pids = num_labeled_pids # 482 for PRW
        self.cq_size = cq_size
        self.momentum = momentum
        self.scalar = scalar
        self.size_average = size_average

        self.fc = nn.Linear(256, num_labeled_pids)

        # Unlabeled targets = 5555
    # softmax
    def forward(self, inputs, targets, iteration):
        labeled = (targets >= 0)
        inputs = inputs[labeled]
        if len(inputs) == 0:
            return inputs.sum(), inputs
        inputs = self.fc(inputs/self.scalar)
        loss = F.cross_entropy(inputs, targets[labeled], size_average=self.size_average)
        return loss, inputs


class OIM(autograd.Function):

    def __init__(self, lut, cq, header, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.cq = cq
        self.header = header
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([self.lut, self.cq], dim=0))

        for x, y in zip(inputs, targets):
            if y < len(self.lut) and y >= 0:
                self.lut[y] = self.momentum * \
                    self.lut[y] + (1. - self.momentum) * x
                self.lut[y] /= self.lut[y].norm()
            elif y >= len(self.lut):
                self.cq[self.header] = x
                self.header = (self.header + 1) % self.cq.size(0)

        return grad_inputs, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM(lut, cq, header, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_labeled_pids, cq_size=5000, scalar=10.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_labeled_pids = num_labeled_pids
        self.cq_size = cq_size
        self.momentum = momentum
        self.scalar = scalar
        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(
            (self.num_labeled_pids, self.num_features)))
        self.register_buffer('cq',  torch.zeros(
            (self.cq_size, self.num_features)))

        self.header_cq = 0
        self.weight = torch.cat([torch.ones(self.num_labeled_pids),
                                 torch.zeros(self.cq_size)]).cuda()
        ## Unlabeled targets = 5555

    def forward(self, inputs, targets, iteration=0):
        unlab = targets < 0
        targets[unlab] =  5555 #5555

        inputs = oim(inputs, targets, self.lut, self.cq,
                     self.header_cq, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               size_average=self.size_average,
                               ignore_index=5555)

        self.header_cq = ((self.header_cq + (targets >= len(self.lut)
                                             ).long().sum().item()) % self.cq.size(0))
        return loss, inputs
        

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()