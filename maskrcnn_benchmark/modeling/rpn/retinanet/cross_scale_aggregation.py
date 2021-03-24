import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.nn.parameter import Parameter

from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.layers import Scale

import math
import numpy as np

class CrossScaleAggregationModule(nn.Module):

	# Cross Scale Aggregation

	def __init__(self, in_channels, points_feat_dim=256, n_anchor=1):
		super(CrossScaleAggregationModule, self).__init__()

		self.start_level = 3 # RetinaNet range from 3 to 7

		self.query_fc = nn.Sequential(
			nn.Linear(points_feat_dim, in_channels)
		)
		
		self.key_fc = nn.Sequential(
			nn.Linear(points_feat_dim, in_channels)
		)

		self.n_anchor = n_anchor

		self.scale3 = nn.Parameter(torch.FloatTensor([1.]))
		self.scale4 = nn.Parameter(torch.FloatTensor([1.]))
		self.scale5 = nn.Parameter(torch.FloatTensor([1.]))
		self.scale6 = nn.Parameter(torch.FloatTensor([1.]))
		self.scale7 = nn.Parameter(torch.FloatTensor([1.]))
		self.scales = [self.scale3, self.scale4, self.scale5, self.scale6, self.scale7]

	def forward(self, pyramids, boxes, points_feat, centers, box_feat):
		# boxes, [(selected_pos_img1, 4), ...]
		
		# points_feat, feature of selected positive points
		# [(selected_pos_img1, feat_dim), ...]

		# centers, 3dim, (y, x, lvl)
		# [[centers for lvl1 in img1, ..., centers for lvlL in img1], ...]

		# box_feat, (N, h1w1+h2w2+..., feat_dim)

		# return, (selected_all_imgs, feat_dim)

		assert len(boxes) == len(centers)
		n_per_img = [f.size(0) for f in points_feat]
		query_points_feat = self.query_fc(torch.cat(points_feat, dim=0))
		query_points_feat = query_points_feat.split(n_per_img, dim=0) # [(selected_pos_img1, feat_dim), ...]
		centers = [torch.cat(center_list, dim=0) for center_list in centers]

		results = []
		for center_per_img, box_per_img, points_feat_per_img, cand_feat_per_img, query_per_img in zip(
			centers, boxes, points_feat, box_feat, query_points_feat):
			coord_in_img = center_per_img[:, :2] + center_per_img[:, [2]]//2 # (h1w1+h2w2+..., 2) or (all, 2)

			if not self.training and box_per_img.size(0) == 0:
				continue

			stride = center_per_img[:, -1]
			lvl = torch.log2(stride.float()) - self.start_level
			lvl = lvl.to(torch.long)

			xs, ys = coord_in_img[:, 1], coord_in_img[:, 0] # (all, )

			l = xs[:, None] - box_per_img[:, 0][None] # (all, box_in_img)
			t = ys[:, None] - box_per_img[:, 1][None]
			r = box_per_img[:, 2][None] - xs[:, None]
			b = box_per_img[:, 3][None] - ys[:, None]

			cand_in = torch.stack([l, t, r, b], dim=2) # (all, box_in_img, 4)
			cand_in = cand_in.min(dim=2)[0] > 0 # (all, box_in_img)

			for idx, (per_points_feat, per_points_query) in enumerate(zip(points_feat_per_img, query_per_img)):
				per_cand_feat = cand_feat_per_img[cand_in[:, idx], :] # (sel_cand, feat_dim)

				lvl_per = lvl[cand_in[:, idx]] # (sel_cand, )

				scales = torch.cat(self.scales, dim=0)
				weight = scales[lvl_per] # (sel_cand, )
				
				key_per_cand_feat = self.key_fc(per_cand_feat*weight.unsqueeze(-1))
				sim = torch.matmul(key_per_cand_feat, per_points_query) # (sel_cand, )
				sim = sim.clamp(min=-50, max=50)
				sim = F.softmax(sim)
				merge = (sim.unsqueeze(-1) * per_cand_feat).sum(dim=0) # (feat_dim, )
				merge = per_points_feat + merge
				results.append(merge)

		return torch.stack(results, dim=0)


class CrossScaleAggregationModuleMultiAnchor(CrossScaleAggregationModule):

	# Cross Scale Aggregation for multiple anchors

	def __init__(self, in_channels, points_feat_dim=256, n_anchor=1):
		super(CrossScaleAggregationModuleMultiAnchor, self).__init__(in_channels, points_feat_dim, n_anchor)


	def forward(self, pyramids, boxes, points_feat, centers, box_feat):

		assert len(boxes) == len(centers)
		n_per_img = [f.size(0) for f in points_feat]
		query_points_feat = self.query_fc(torch.cat(points_feat, dim=0))
		query_points_feat = query_points_feat.split(n_per_img, dim=0)
		centers = [torch.cat(center_list, dim=0) for center_list in centers]

		
		results = []
		for center_per_img, box_per_img, points_feat_per_img, cand_feat_per_img, query_per_img in zip(
			centers, boxes, points_feat, box_feat, query_points_feat):
			coord_in_img = center_per_img[:, :2] + center_per_img[:, [2]]//2 # (all, 2)

			coord_in_img = coord_in_img.view(-1, self.n_anchor, 2)
			coord_in_img = coord_in_img[:, 0, :] # (all/n_anchor, 2)
			cand_feat_per_img = cand_feat_per_img.view(coord_in_img.size(0), self.n_anchor, -1)
			cand_feat_per_img = cand_feat_per_img.mean(dim=1) # (all/n_anchor, feat_dim)

			stride = center_per_img[:, -1]
			lvl = torch.log2(stride.float()) - self.start_level
			lvl = lvl.to(torch.long)		
			lvl = lvl.view(coord_in_img.size(0), self.n_anchor)
			lvl = lvl[:, 0]

			xs, ys = coord_in_img[:, 1], coord_in_img[:, 0] # (all, )

			l = xs[:, None] - box_per_img[:, 0][None] # (all, box_in_img)
			t = ys[:, None] - box_per_img[:, 1][None]
			r = box_per_img[:, 2][None] - xs[:, None]
			b = box_per_img[:, 3][None] - ys[:, None]

			cand_in = torch.stack([l, t, r, b], dim=2) # (all, box_in_img, 4)
			cand_in = cand_in.min(dim=2)[0] > 0 # (all, box_in_img)

			for idx, (per_points_feat, per_points_query) in enumerate(zip(points_feat_per_img, query_per_img)):
				per_cand_feat = cand_feat_per_img[cand_in[:, idx], :]
				lvl_per = lvl[cand_in[:, idx]]

				scales = torch.cat(self.scales, dim=0)
				weight = scales[lvl_per]
				
				key_per_cand_feat = self.key_fc(per_cand_feat*weight.unsqueeze(-1))
				sim = torch.matmul(key_per_cand_feat, per_points_query)
				sim = sim.clamp(min=-30, max=30)
				sim = F.softmax(sim)
				merge = (sim.unsqueeze(-1) * per_cand_feat).sum(dim=0)
				merge = per_points_feat + merge
				results.append(merge)

		return torch.stack(results, dim=0)


def box_iou(box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # (N,M,2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N,M)

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def make_cross_scale_aggregation_module(cfg, in_channels):
	n_anchor = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
	points_feat_dim = cfg.MODEL.REID_CONFIG.FEAT_DIM
	if n_anchor == 1:
		return CrossScaleAggregationModule(in_channels, points_feat_dim)
	else:
		return CrossScaleAggregationModuleMultiAnchor(in_channels, points_feat_dim, n_anchor)
