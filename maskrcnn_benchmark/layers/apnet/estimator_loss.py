import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.bounding_box import BoxList

class EstLoss(nn.Module):
	def __init__(self, loss="smooth_l1", bidirection=False):
		super(EstLoss, self).__init__()
		if loss == "smooth_l1":
			self.loss = smooth_l1_loss
		elif loss == "l2":
			self.loss = l2_loss

		self.encoder = est_encode if not bidirection else est_encode_bidirection

	def forward(self, reg_vals, reg_gts):
		if reg_gts.dim() > 1:
			eff = ~(reg_gts[:, 1]==1)
			return self.loss(reg_vals[eff, :], self.mapping(reg_gts[eff, :]))
		else:
			eff = ~(reg_gts==1)
			return self.loss(reg_vals[eff], self.mapping(reg_gts[eff]))

	def mapping(self, reg_gts):
		# mapped to [-1, 1]
		return self.encoder(reg_gts)



def est_encode(reg_gts):
	return 2*reg_gts - 1

def est_decode(est_vals):
	return (est_vals+1) / 2

def est_encode_bidirection(reg_gts):
	# reg_gts[:, 0], [-0.2, 0.2]
	# reg_gts[:, 1], [-0.3, 0.7]
	return torch.cat([reg_gts[:, [0]]/0.2, 2*reg_gts[:, [1]]-0.4], dim=1)

def est_decode_bidirection(est_vals):
	return torch.cat([est_vals[:, [0]]*0.2, (est_vals[:, [1]]+0.4)/2], dim=1)

def est_decode_regval2proposal(proposals, add_padratio=False):
	return_bboxlist = []
	for proposal in proposals:
		p_bbox = 1.0*proposal.bbox
		new_bbox = []
		n_proposal = p_bbox.shape[0]

		regvals = proposal.get_field("reg_vals")
		reg_vals = est_decode(regvals)
                
		for j in range(n_proposal):
			bbox = p_bbox[j, :]
			h = bbox[3] - bbox[1]
			new_h = h*(1./(1.-reg_vals[j]))
			bbox[3] = bbox[1] + new_h
			new_bbox.append(bbox.tolist())

		if n_proposal == 0:
			new_bbox = torch.tensor([]).view(0, 4)
		new_bboxlist = BoxList(new_bbox, proposal.size, mode="xyxy")
		new_bboxlist._copy_extra_fields(proposal)
		if add_padratio:
			new_bboxlist.add_field("pad_ratio", reg_vals)
		return_bboxlist.append(new_bboxlist)

	return return_bboxlist

def est_decode_bidirection_regval2proposal(proposals, add_padratio=False):
	return_bboxlist = []

	for proposal in proposals:
		p_bbox = 1.0*proposal.bbox
		new_bbox = []
		n_proposal = p_bbox.shape[0]

		regvals = proposal.get_field("reg_vals")
		reg_vals = est_decode_bidirection(regvals)
                
		for j in range(n_proposal):
			bbox = p_bbox[j, :]
			h = bbox[3] - bbox[1]
			ratio = h*(1./(1.-reg_vals[j, 0]-reg_vals[j, 1]))
			bbox[1] -= ratio * reg_vals[j, 0]
			bbox[3] += ratio * reg_vals[j, 1]
			new_bbox.append(bbox.tolist())

		if n_proposal == 0:
			new_bbox = torch.tensor([]).view(0, 4)
		new_bboxlist = BoxList(new_bbox, proposal.size, mode="xyxy")
		new_bboxlist._copy_extra_fields(proposal)
		if add_padratio:
			new_bboxlist.add_field("pad_ratio", reg_vals)
		return_bboxlist.append(new_bboxlist)

	return return_bboxlist






def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):

    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def l2_loss(input, target, size_average=True):
	loss = torch.pow(input-target, 2)
	if size_average:
		return loss.mean()
	return loss.sum()