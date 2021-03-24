import torch
import torch.nn.functional as F
import numpy as np

from maskrcnn_benchmark.layers.apnet.estimator_loss import est_decode_regval2proposal, est_decode_bidirection_regval2proposal
from maskrcnn_benchmark.structures.bounding_box import BoxList


class Ratio(object): 
	def __init__(self, discard_nokp=False):

		self.thrx_idx = 7
		self.rhip_idx = 2
		self.lhip_idx = 3
		self.rknee_idx = 1
		self.lknee_idx = 4
		self.rank_idx = 0
		self.lank_idx = 5

		self.r_head2thrx = 0.24
		self.r_thrx2hip = 0.32
		self.r_hip2knee = 0.22
		self.r_knee2ank = 0.21


		self._rat2 = [1., self.r_thrx2hip/self.r_head2thrx, self.r_hip2knee/self.r_head2thrx, self.r_knee2ank/self.r_head2thrx]
		self._pratio = [self.r_head2thrx, self.r_thrx2hip, self.r_hip2knee, self.r_knee2ank, 0.]
		self._idx = [(self.thrx_idx, ), (self.rhip_idx, self.lhip_idx), (self.rknee_idx, self.lknee_idx), (self.rank_idx, self.lank_idx)]
		
		
		self.INVIS_THRSH = 0.1
		self.discard_nokp = discard_nokp

	def get_ratio_by_est(self, boxlist, is_train):


		device = boxlist[0].bbox.device 

		"""
		return_boxlist = []
		for target in boxlist:
			target_bbox = target.bbox
			n = target_bbox.shape[0]
			img_size = target.size
			regvals = target.get_field("reg_vals")
			reg_vals = est_decode(regvals)
			
			new_bbox = []
			for k in range(n):
				p_bbox = target_bbox[k, :]
				h = p_bbox[3]-p_bbox[1]
				new_h = h*(1./(1.-reg_vals[k]))
				p_bbox[3] = p_bbox[1] + new_h
				new_bbox.append(p_bbox.tolist())

			new_bboxlist = BoxList(new_bbox, img_size, mode="xyxy")
			new_bboxlist._copy_extra_fields(target)
			new_bboxlist.add_field("pad_ratio", reg_vals)

			return_boxlist.append(new_bboxlist)
		"""
		return_boxlist = est_decode_regval2proposal(boxlist, add_padratio=True)

		return_boxlist = [return_box.to(device) for return_box in return_boxlist]

		return return_boxlist


	def get_ratio(self, boxlist, is_train):
		"""

			boxlist: [Bbox, Bbox, ...]

		"""	
		"""
			for those without keypoints:
				global, not partition
		
		"""
		return_boxlist = []
		device = boxlist[0].bbox.device

		for target in boxlist:
			target_bbox = target.bbox
			keypoint = target.get_field("keypoints")
			kp = keypoint.keypoints
			n, _, _ = kp.shape
			bbox = target.bbox
			img_size = target.size

			new_bbox = []
			new_pad = []
			for k in range(n):
				p_kp = kp[k]
				
				pad = 1.0 if is_train else 0.
			
				if p_kp.sum().item() > 0:
					pad = 0.
					for iteration, i in enumerate(self._idx[::-1][:-1]):
						# assume thorax exists
						vis = False
						store_y = None
						for j in i:
							if p_kp[j][2] > self.INVIS_THRSH:
								vis = True

						if vis: 
							store_y = max(p_kp[i[0]][1], p_kp[i[1]][1])
							break



					if not vis:
						# hips, knees, ankles not visible
						pad += sum(self._pratio[2:])
						res = F.relu(target_bbox[k, 3]-p_kp[self.thrx_idx, 1])
						known = F.relu(p_kp[self.thrx_idx, 1]-target_bbox[k, 1])
						tmp = F.relu((self.r_thrx2hip/self.r_head2thrx)*known-res) # pixel
						pad += self.r_thrx2hip*tmp/(tmp+res).item()

						if p_kp[self.thrx_idx, 1].item() == 0:
							pad = 1.0
				
					elif iteration == 0:
						pad = 0.
				
					else:
						pad += sum(self._pratio[::-1][:iteration])
						res = F.relu(target_bbox[k, 3]-store_y)
						known = F.relu(p_kp[self.thrx_idx, 1]-target_bbox[k, 1])
						tmp = F.relu((self._pratio[::-1][iteration]/self.r_head2thrx)*known-res)
						pad += (self._pratio[::-1][iteration]*tmp/(tmp+res)).item()
					
						if p_kp[self.thrx_idx, 1].item() == 0:
							pad = 1.0
							

				p_bbox = 1.*bbox[k, :]
				h = p_bbox[3] - p_bbox[1]
				if pad == 1.0:
					new_h = h
					if not is_train:
						pad = 0.
				else:
					new_h = h*(1./(1.-pad))
				p_bbox[3] = p_bbox[1] + new_h
				new_bbox.append(p_bbox.tolist())
				new_pad.append(pad)



			new_bboxlist = BoxList(new_bbox, img_size, mode="xyxy")
			new_bboxlist._copy_extra_fields(target)
			new_bboxlist.add_field("pad_ratio", torch.tensor(new_pad))
			return_boxlist.append(new_bboxlist)

		return_boxlist = [return_box.to(device) for return_box in return_boxlist]


		return return_boxlist




class Ratio_Bidirection(object): 
	def __init__(self, discard_nokp):


		self.neck_idx = 8
		self.head_idx = 9
		self.thrx_idx = 7
		self.rhip_idx = 2
		self.lhip_idx = 3
		self.rknee_idx = 1
		self.lknee_idx = 4
		self.rank_idx = 0
		self.lank_idx = 5


		self.r_head2neck = 0.13
		self.r_neck2thrx = 0.06
		self.r_thrx2hip = 0.32
		self.r_hip2knee = 0.22
		self.r_knee2ank = 0.21


		self._rat2 = [1., self.r_thrx2hip/self.r_neck2thrx, self.r_hip2knee/self.r_neck2thrx, self.r_knee2ank/self.r_neck2thrx]
		self._pratio = [self.r_neck2thrx, self.r_thrx2hip, self.r_hip2knee, self.r_knee2ank, 0.]
		self._idx = [(self.thrx_idx, ), (self.rhip_idx, self.lhip_idx), (self.rknee_idx, self.lknee_idx), (self.rank_idx, self.lank_idx)]
		
		
		"""
			for example, if knee is missing: 
				add the knee2ank part
	
		"""
		self.INVIS_THRSH = 0.1
		self.discard_nokp = discard_nokp


	def get_ratio_query(self, boxlist, is_train):
		device = boxlist[0].bbox.device 
		return_boxlist = est_decode_bidirection_regval2proposal(boxlist, add_padratio=True)
		return_boxlist = [return_box.to(device) for return_box in return_boxlist]

		return return_boxlist



	def get_ratio(self, boxlist, is_train):
		"""

			boxlist: [Bbox, Bbox, ...]

		"""	
		"""
			for those without keypoints:
				global, not partition
		
		"""
		return_boxlist = []
		device = boxlist[0].bbox.device

		for target in boxlist:
			target_bbox = target.bbox

			# store_bbox = 1.*target_bbox


			keypoint = target.get_field("keypoints")
			kp = keypoint.keypoints
			n, _, _ = kp.shape
			bbox = target.bbox
			img_size = target.size

			new_bbox = []
			new_pad = []
			new_uppad = []

			for k in range(n):
				p_kp = kp[k]

				pad = 1.0 if is_train else 0.

				uppad = 0.


				if p_kp.sum().item() > 0:
					

					if p_kp[self.head_idx, 2] > self.INVIS_THRSH:
						
						h_head2neck = p_kp[self.neck_idx, 1] - p_kp[self.head_idx, 1]
						up_blank = p_kp[self.head_idx, 1] - target_bbox[k, 1]
						if up_blank > 0.8*h_head2neck:
							h_neck2thrx = p_kp[self.thrx_idx, 1] - p_kp[self.neck_idx, 1]

							if h_neck2thrx != 0:
								uppad = F.relu((up_blank/h_neck2thrx)*self.r_neck2thrx)
								uppad = (uppad*(-1)).item()
							else:
								uppad = 0.



						else:
							uppad = 0.

					
					else:
						# head partial invisible
						res = F.relu(p_kp[self.neck_idx, 1]-target_bbox[k, 1])
						known = F.relu(p_kp[self.thrx_idx, 1]-p_kp[self.neck_idx, 1])
						tmp = F.relu((self.r_head2neck/self.r_neck2thrx)*known-res)
						if tmp+res == 0:
							uppad = 0.
						else:
							uppad += (self.r_head2neck*tmp/(tmp+res)).item()




				if p_kp.sum().item() > 0:
					pad = 0.
					for iteration, i in enumerate(self._idx[::-1][:-1]):
						# assume thorax exists
						vis = False
						store_y = None
						for j in i:
							if p_kp[j][2] > self.INVIS_THRSH:
								vis = True

						if vis: 
							store_y = max(p_kp[i[0]][1], p_kp[i[1]][1])
							break



					if not vis:
						# hips, knees, ankles not visible
						# only thorax visible

						pad += sum(self._pratio[2:])
						res = F.relu(target_bbox[k, 3]-p_kp[self.thrx_idx, 1])
						known = F.relu(p_kp[self.thrx_idx, 1]-p_kp[self.neck_idx, 1])

						tmp = F.relu((self.r_thrx2hip/self.r_neck2thrx)*known-res) # pixel
						


						"""
						if p_kp[self.thrx_idx, 1].item() == 0:
							pad = 1.0
						"""
						if tmp+res == 0:
							pad = 1.0
						else:
							pad += (self.r_thrx2hip*tmp/(tmp+res)).item()

				

					elif iteration == 0:
						# ankle visible
						
						h_neck2thrx = p_kp[self.thrx_idx, 1] - p_kp[self.neck_idx, 1]
						bottom_blank = target_bbox[k, 3] - store_y
			
						refr = max(p_kp[self.rank_idx, 1], p_kp[self.lank_idx, 1]) - max(p_kp[self.rknee_idx, 1], p_kp[self.lknee_idx, 1])

						if bottom_blank > refr and h_neck2thrx != 0:
							pad = (bottom_blank/h_neck2thrx)*self.r_neck2thrx
							pad = (pad*(-1)).item()
							pad -= pad*0.3
						else:
							pad = 0.
						
						
						

				
					else:
						pad += sum(self._pratio[::-1][:iteration])
						res = F.relu(target_bbox[k, 3]-store_y)
						known = F.relu(p_kp[self.thrx_idx, 1]-p_kp[self.neck_idx, 1])
						tmp = F.relu((self._pratio[::-1][iteration]/self.r_neck2thrx)*known-res)

						if tmp+res == 0.:
							pad = 1.0
						else:
							pad += (self._pratio[::-1][iteration]*tmp/(tmp+res)).item()
						
						
						
				
				p_bbox = 1.*bbox[k, :]
				h = p_bbox[3] - p_bbox[1]

				if pad == 1.0:
					if not is_train:
						pad = 0.
				elif 1.-pad-uppad <= 0:
					pad = 1.0
					if not is_train:
						pad = 0.
				else:

					eff_strip = h*(1./(1.-pad-uppad))

					p_bbox[1] = p_bbox[1] - eff_strip*uppad
					p_bbox[3] = p_bbox[3] + eff_strip*pad

		
				new_bbox.append(p_bbox.tolist())
				new_pad.append(pad)
				new_uppad.append(uppad)



			uppad = torch.tensor(new_uppad).unsqueeze(-1)
			pad = torch.tensor(new_pad).unsqueeze(-1)
			
			new_bboxlist = BoxList(new_bbox, img_size, mode="xyxy")
			new_bboxlist._copy_extra_fields(target)
			new_bboxlist.add_field("pad_ratio", torch.cat([uppad, pad], dim=1))
			return_boxlist.append(new_bboxlist)

		return_boxlist = [return_box.to(device) for return_box in return_boxlist]


		return return_boxlist