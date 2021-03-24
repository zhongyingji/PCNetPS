import torch

from ..utils import permute_and_flatten, l2norm

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from .inference import RetinaNetPostProcessor


class RetinaNetPostProcessorDrawPoints(RetinaNetPostProcessor):
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        box_coder=None,
        query_ovlp_thsh=0.5,
        query_ovlp_anc=True,
        query_ovlp_topk=32
    ):
        super(RetinaNetPostProcessorDrawPoints, self).__init__(
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            min_size,
            num_classes,
            box_coder,
            query_ovlp_thsh,
            query_ovlp_anc,
            query_ovlp_topk
        )

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression, box_offset, box_dconv_feat, 
        centers, fpn_lvl, 
        targets, query=False):

        assert (targets is None) ^ query

        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A
        FEAT_DIM = box_dconv_feat.size(1) // A

        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        n_points = box_offset.size(1) // 2
        box_offset = permute_and_flatten(box_offset, N, A, 2*n_points, H, W)

        box_feat = permute_and_flatten(box_dconv_feat, N, A, FEAT_DIM, H, W)
        box_feat = box_feat.reshape(N, -1, FEAT_DIM)

        if query:
            query_candidates = anchors
            if self.q_ovlp_anc:
                query_candidates = [q_cand.clip_to_image(remove_empty=False) for q_cand in query_candidates]
            else:
                # overlap by detected bounding boxes
                detections = []
                for regression, anchor in zip(box_regression, anchors):
                    detection = self.box_coder.decode(regression, anchor.bbox)
                    boxlist = BoxList(detection, anchor.size, mode="xyxy")
                    boxlist = boxlist.clip_to_image(remove_empty=False)
                    detections.append(boxlist)
                query_candidates = detections
            for q_cand, feat in zip(query_candidates, box_feat):
                q_cand.add_field("points_feats", feat)
                    
            return query_candidates

        num_anchors = A * H * W

        candidate_inds = box_cls > self.pre_nms_thresh

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        results = []
        for per_box_cls, per_box_regression, per_box_feat, per_loc, per_offset, \
        per_pre_nms_top_n, per_candidate_inds, per_anchors in zip(
            box_cls,
            box_regression,
            box_feat, 
            centers, 
            box_offset, 
            pre_nms_top_n,
            candidate_inds,
            anchors):

            per_box_cls = per_box_cls[per_candidate_inds]
 
            per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = \
                    per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]
            per_class += 1

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            per_box_feature = per_box_feat[per_box_loc, :].view(-1, FEAT_DIM)
            per_box_center = per_loc[per_box_loc, :].view(-1, 3)
            per_box_center = per_box_center[:, :-1]
            per_box_offset = per_offset[per_box_loc, :].view(-1, 2*n_points)

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("points_feats", per_box_feature)

            boxlist.add_field("centers", per_box_center)
            boxlist.add_field("offsets", per_box_offset)
            boxlist.add_field("fpn_lvls", fpn_lvl * torch.ones_like(per_class).to(per_class.device))

            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, anchors, centers, objectness, box_regression, 
        box_offset, features, 
        reid_pool, targets, query=False, reid_feat=None, reid_bn=None):
       
        centers = list(zip(*centers))
        start_level = 3 # RetinaNet start level

        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        A = box_regression[0].size(1) // 4

        self.reid_feat_func = reid_pool["reid_feat_func"]
        self.reid_feat_func_pooling = reid_pool["reid_feat_func_pooling"]

        if reid_feat is None:
            box_dconv_feat = self.reid_feat_func(features, box_offset, centers)
        else:
            box_dconv_feat = reid_feat

        for lvl, (a, o, b, ofs, f, ctr) in enumerate(zip(anchors, objectness, box_regression, box_offset, box_dconv_feat, centers), start_level):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b, ofs, f, ctr, lvl, targets, query))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if query:
            return self.query_composition(targets, boxlists, features, centers, reid_bn)
        else:
            if num_levels > 1:
                boxlists = self.select_over_all_levels(boxlists)

            len_boxes = [len(boxlist) for boxlist in boxlists]
            points_feat = [boxlist.get_field("points_feats") for boxlist in boxlists]

            box_dconv_feat_flat_N = []
            for per_dconv_feat in box_dconv_feat:
                n, Axfeat_dim, h, w = per_dconv_feat.shape
                feat_dim = Axfeat_dim // A
                box_dconv_feat_flat_N.append(permute_and_flatten(per_dconv_feat, n, A, feat_dim, h, w))
            box_dconv_feat_flat_N = torch.cat(box_dconv_feat_flat_N, dim=1)
            

            box_feat = points_feat
            """
            box_feat = self.reid_feat_func_pooling(features, 
                [boxlist.bbox for boxlist in boxlists], 
                points_feat, 
                centers,
                box_dconv_feat_flat_N)
            box_feat = box_feat.split(len_boxes)
            """

            for per_boxlist, per_feat in zip(boxlists, box_feat):
                if per_feat.size(0) != 0:
                    per_feat = l2norm(reid_bn(per_feat))
                per_boxlist.add_field("feats", per_feat)

        return boxlists


    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            feats = boxlists[i].get_field("points_feats")
            ctrs = boxlists[i].get_field("centers")
            offsets = boxlists[i].get_field("offsets")
            fpn_lvls = boxlists[i].get_field("fpn_lvls")

            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                feats_j = feats[inds]
                boxes_j = boxes[inds, :].view(-1, 4)

                ctrs_j = ctrs[inds, :].view(-1, 2)
                offsets_j = offsets[inds, :]
                fpn_lvls_j = fpn_lvls[inds]

                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class.add_field("points_feats", feats_j)
                boxlist_for_class.add_field("centers", ctrs_j)
                boxlist_for_class.add_field("offsets", offsets_j)
                boxlist_for_class.add_field("fpn_lvls", fpn_lvls_j)

                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
