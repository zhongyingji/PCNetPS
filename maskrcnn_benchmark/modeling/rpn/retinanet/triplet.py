import torch
import torch.nn as nn
import torch.nn.functional as F

class onlineTriplet(nn.Module):
	def __init__(self, margin):
		super(onlineTriplet, self).__init__()
		self.margin = margin

	def forward(self, anc_feat, pos_feat, ids):
		# anc_feat, (n, feat_dim), attentioned feat
		# pos_feat, (n, feat_dim), vanilla feat

		dist_ap = (anc_feat * pos_feat).sum(dim=1) # (n, )

		"""for each attentioned feat, select the nearest attentioned feat with other ids
		"""

		mask = self.dist_vec(ids)
		mask = (mask == 0) # (n, n)

		dist_aa = torch.matmul(anc_feat, anc_feat.t()) # (n, n)
		dist_aa[mask] = -10
		dist_an, idx_an = dist_aa.max(dim=1) # (n, )

		return F.relu(self.margin+dist_an-dist_ap).mean()

	def dist_vec(self, vector):
		vector = vector.view(-1, 1).float()
		product = -2*torch.matmul(vector, torch.t(vector))
		sq_1 = torch.pow(vector, 2).view(1, -1)
		sq_2 = torch.pow(vector, 2).view(-1, 1)
		return sq_1+sq_2+product

