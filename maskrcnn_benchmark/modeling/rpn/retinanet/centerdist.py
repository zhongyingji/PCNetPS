import torch
import torch.nn as nn

class Centerdist(nn.Module):
	def __init__(self):
		super(Centerdist, self).__init__()
	def forward(self, reid_feat, ids):
		uniq_ids = torch.unique(ids)
		n_uniq = len(uniq_ids)
		loss = []
		for per_id in uniq_ids:
			if per_id == -2:
				continue
			idx = torch.nonzero(ids == per_id).squeeze(1)
			per_feat = reid_feat[idx, :]
			# (n, feat_dim)
			n_curid = per_feat.size(0)
			mean_feat = per_feat.mean(dim=0)
			# (feat_dim, )
			loss.append(torch.pow(per_feat - mean_feat.unsqueeze(0), 2).sum() / n_curid)

		return sum(loss) / n_uniq
