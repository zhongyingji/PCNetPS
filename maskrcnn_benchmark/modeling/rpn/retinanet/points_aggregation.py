import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.nn.parameter import Parameter

from maskrcnn_benchmark.layers.misc import DFConv2d_Nonoffset as DFConv2d

import math
import numpy as np

class PointsAggregationModule(nn.Module):

	# Contextual Points Pooling

	def __init__(self, 
		in_channels, 
		pyramid_level, 
		num_points, 
		num_group_points, 
		group_feat_concat, 
		loss_between_offsets, 
		sets_per_group, 
		points_feat_dim=256
		):
		super(PointsAggregationModule, self).__init__()

		self.pyramid_level = pyramid_level
		self.n_points = num_points
		self.n_group_points = num_group_points
		self.group_feat_concat = group_feat_concat
		self.loss_between_offsets = loss_between_offsets
		self.sets_per_group = sets_per_group

		self.dfconv2d = nn.Sequential(
			*[nn.ModuleList() for _ in range(self.n_group_points)])

		self.offset_conv = nn.ModuleList()
		
		out_channels = points_feat_dim

		if self.n_group_points > 1 and self.group_feat_concat:
			# if group is 1
			# simply add the loss on mass
			assert out_channels % self.n_group_points == 0
			out_channels = out_channels // self.n_group_points

		kernel_size = int(math.sqrt(self.n_points))
		self.start_level = 3 # for RetinaNet, the FPN level ranges from 3 to 7

		for i in range(self.pyramid_level):
			for g in range(self.n_group_points):
				self.dfconv2d[g].append(
					DFConv2d(
						in_channels, 
						out_channels, 
						kernel_size=kernel_size,
						stride=1, 
						groups=1, 
						dilation=1, 
						deformable_groups=1, 
						bias=False
					)
				)
				
		self.eps = 1e-8

	def forward_per_group(self, group_idx, pyramid, offset, centers):
		# offset, [(n, n_anchor, 2*n_points, h, w), ...]
		# centers, [(h1w1*3, h1w1*3, ...), (h2w2*3, h2w2*3, ...), ...]
		results = []
		offset_updim = []

		for (offset_lvl, per_offset), per_center in zip(enumerate(offset), centers):
			n, n_anchor, _, oh, ow = per_offset.shape
			per_offset_flat = per_offset.view(n*n_anchor, -1, oh, ow) # (n*n_anchor, 2*n_points, oh, ow)
			per_pyramid = pyramid[offset_lvl]
			per_pyramid = per_pyramid.repeat(1, n_anchor, 1, 1).view(n*n_anchor, -1, oh, ow) # (n*n_anchor, points_dim, oh, ow)
			dconv_op = self.dfconv2d[group_idx][offset_lvl]
			result_per_lvl = dconv_op(per_pyramid, per_offset_flat) # (n*n_anchor, points_dim, oh, ow)
			results.append(result_per_lvl)
			
		return results # [(n*n_anchor, points_dim, oh, ow), ...]


	def forward(self, pyramid, offset, centers):
		# pyramid, [nchw, ...]
		# offset, [(n, 2*n_points*dconv_level*n_anchor, h, w), ...]

		# for centers
		# [[f1*w1*3, f2*w2*3, ...], ...]
		#   \                    /
		#     all centers of one image
		#                   \        /
		#                    all images

		centers = list(zip(*centers))

		n, c, _, _ = offset[0].shape
		n_anchor = c // self.n_points // self.n_group_points // 2

		offset_view = [per_offset.view(n, n_anchor, self.n_group_points, -1, 
			per_offset.size(2), per_offset.size(3))
			for per_offset in offset]
		# [(n, n_anchor, groups, 2*n_points*dconv_lvl, h, w), ...]
	
		results = []
		for g in range(self.n_group_points):
			offset_per_group = [
				per_offset[:, :, g]
				for per_offset in offset_view
			]
			
			results.append(self.forward_per_group(g, 
				pyramid, offset_per_group, centers))

		# [[nch1w1, nch2w2, nch3w3, ...], [nch1w1, nch2w2, ...]]
		# combining the groups
		results = list(zip(*results))
		if self.group_feat_concat:
			results = [torch.cat(result, dim=1).view(
				n, -1, result[0].size(2), result[0].size(3)
				) # (n, n_anchor*points_dim, h, w)
				for result in results]
		else:
			results = [torch.stack(result, dim=0).sum(dim=0).view(
				n, -1, result[0].size(2), result[0].size(3)
				)
				for result in results]
		# [nch1w1, nch2w2, ...]


		return results


	# Distance loss for scattering the points
	def get_points_dist_loss(self, centers, offsets, regression_targets):
		# centers, (n, 3)
		# offsets, (n, groups*n_points*2)
		# regression_targets, (n, 4)

		losses = None

		if not self.loss_between_offsets:
			return None

		else:
			N = centers.size(0)
			points_stride = centers[:, -1]
			points_lvl = torch.log2(points_stride).long()
			offsets_g = offsets * (points_stride.unsqueeze(-1)) # note: into image space

			offsets_g = offsets_g.view(N, self.n_group_points, 1, -1) # (n, groups, 1, 2*n_points)
			
			if self.n_group_points == 1:
				offsets_g = offsets_g.squeeze(1).view(N, -1, self.n_points, 2) # (n, 1, n_points, 2)

				offsets_m = self.get_setdist(offsets_g) # [(n, 1), (n, 1), ...]

				losses = []
				h_bbox = (regression_targets[:, 3] - regression_targets[:, 1]).clone().detach() # (n, )
			
				h_bbox = h_bbox // self.sets_per_group
				h_bbox = h_bbox.unsqueeze(-1) # (n, 1)
				for per_offset_m in offsets_m:
					losses.append((torch.pow(h_bbox - per_offset_m, 2)/h_bbox+self.eps).sum())
				losses = torch.stack(losses).sum()
				# normalize
				normalisation = float(N * self.n_group_points)
				losses = losses / (max(1, normalisation))

			else:
				offsets_m, offsets_g_tmp = self._groupdist(offsets_g, y_first=True)
				# offsets_m, [(n, 1), (n, 1), ...]
				# offsets_g_tmp, (n, groups, 1, n_points, 2)
				
				setdists = []

				for g in range(offsets_g_tmp.size(1)):
					offsets_g_per_group = offsets_g_tmp[:, g] # (n, 1, n_points, 2)
					offsets_m_per_group = self.get_setdist(offsets_g_per_group)
					setdists.extend(offsets_m_per_group)

				losses = []
				h_bbox = (regression_targets[:, 3] - regression_targets[:, 1]).clone().detach()

				h_bbox_group = (h_bbox // self.n_group_points).unsqueeze(-1) # (n, 1)

				h_bbox_set = (h_bbox // (self.n_group_points * self.sets_per_group)).unsqueeze(-1) # (n, 1)

				for per_offset_m in offsets_m:
					losses.append((torch.pow(h_bbox_group - per_offset_m, 2)/h_bbox_group+self.eps).sum())
				for per_setdist in setdists:
					losses.append((torch.pow(h_bbox_set - per_setdist, 2)/(h_bbox_set+self.eps)).sum())
				losses = torch.stack(losses).sum()
				normalisation = float(N * self.n_group_points * self.sets_per_group)
				losses = losses / (max(1, normalisation))

		return losses



	def get_setdist(self, offsets_g):
		# offsets_g, (n, 1, n_points, 2)

		points_per_sets = self.n_points // self.sets_per_group # 9 // 3

		offsets_g_sets = [
			offsets_g[:, :, (i*points_per_sets):((i+1)*points_per_sets)]
			for i in range(self.sets_per_group-1)
		]

		offsets_g_sets.append(offsets_g[:, :, (self.sets_per_group-1)*points_per_sets:])

		offsets_g_sets = [
			per_offset_g_set.mean(dim=-2) # (n, 1, 2)
			for per_offset_g_set in offsets_g_sets
		]

		offsets_m = self._setdist(offsets_g_sets, y_first=True) # [(n, 1), (n, 1), ...]

		return offsets_m

	def _setdist(self, sets_in_group, y_first=True):
		sets = len(sets_in_group)
		set_dist = [(sets_in_group[i+1] - sets_in_group[i])[:, :, 0 if y_first else 1] # (n, 1, 2)
			for i in range(sets-1)
			]
		return set_dist

	def _groupdist(self, offsets, y_first=True):
		# offsets, (n, groups, 1, 2*n_points)
		
		n, groups, _, points = offsets.size()
		points = points // 2
		offsets = offsets.view(n, groups, 1, points, 2) # (n, groups, 1, n_points, 2)
		offsets_mean = offsets.mean(dim=-2) # (n, groups, 1, 2)
		offsets_mean = [
			(offsets_mean[:, i+1] - offsets_mean[:, i])[:, :, 0 if y_first else 1] # (n, 1, 2)
			for i in range(groups-1)
			]

		return offsets_mean, offsets



def make_points_aggregation_module(cfg, in_channels):
	return PointsAggregationModule(
			in_channels=in_channels, 
			pyramid_level=len(cfg.MODEL.RPN.ANCHOR_SIZES), 
			num_points=cfg.MODEL.PCNET.NUM_POINTS, 
			num_group_points=cfg.MODEL.PCNET.NUM_GROUP_POINTS, 
			group_feat_concat=cfg.MODEL.PCNET.GROUP_CONCAT,
			loss_between_offsets=cfg.MODEL.PCNET.LOSS_BETWEEN_OFFSETS, 
			sets_per_group=cfg.MODEL.PCNET.SETS_PER_GROUP, 
			points_feat_dim=cfg.MODEL.REID_CONFIG.FEAT_DIM)
