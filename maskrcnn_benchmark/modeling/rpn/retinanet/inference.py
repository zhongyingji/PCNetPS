import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten, l2norm

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


class RetinaNetPostProcessor(RPNPostProcessor):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
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
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetPostProcessor, self).__init__(
            pre_nms_thresh, 0, nms_thresh, min_size
        )
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder

        self.q_ovlp_thsh = query_ovlp_thsh
        self.q_ovlp_anc = query_ovlp_anc
        self.q_ovlp_topk = query_ovlp_topk
 
    def add_gt_proposals(self, proposals, targets):
        """
        This function is not used in RetinaNet
        """
        pass

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression, box_dconv_feat, 
        targets, query=False):
        """
        Arguments:
            anchors: list[BoxList] (a single level of all images)
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
            box_dconv_feat: tensor of size N, A * FEAT_DIM, H, W

        Note:
            keep all detected bounding boxes of query
        """

        assert (targets is None) ^ query

        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A
        FEAT_DIM = box_dconv_feat.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

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
        for per_box_cls, per_box_regression, per_box_feat, \
        per_pre_nms_top_n, per_candidate_inds, per_anchors in zip(
            box_cls,
            box_regression,
            box_feat, 
            pre_nms_top_n,
            candidate_inds,
            anchors):

            # Sort and select TopN
            # TODO most of this can be made out of the loop for
            # all images. 
            # TODO:Yang: Not easy to do. Because the numbers of detections are
            # different in each image. Therefore, this part needs to be done
            # per image. 
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

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("points_feats", per_box_feature)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def query_composition(self, gt_boxes, query_candidate, features, centers, reid_bn=None):

        """
            gt_boxes: ground truth of query frames
            query_candidate: ALL anchors/detected bounding boxes as well as their extraction and adjustment
        """

        for image_idx, (gt_box, q_cand, center_per_img) in enumerate(zip(gt_boxes, query_candidate, centers)):
            match_quality_matrix = boxlist_iou(q_cand, gt_box)
            # (n_cand, n_gt)
            assert match_quality_matrix.dim() == 2
            
            ovlp_topk = min(self.q_ovlp_topk, q_cand.bbox.size(0))

            kthval, indices = torch.kthvalue(match_quality_matrix, ovlp_topk, dim=0)
            max_val, max_indices = match_quality_matrix.max(dim=0)
            mat1 = (match_quality_matrix >= kthval)
            mat2 = (match_quality_matrix >= self.q_ovlp_thsh)
            thsh = (mat1*mat2).float()
            # (n_cand, n_gt)
            thsh[max_indices, torch.arange(gt_box.bbox.size(0))] = 1.0
            norm = thsh.sum(dim=0)
            feat = q_cand.get_field("points_feats") # (n_cand, feat_dim)
            q_feat = torch.matmul(thsh.t(), feat)/norm.unsqueeze(-1)

            per_lvl_feature = [feature[[image_idx]] for feature in features]
            box_feat = self.reid_feat_func_pooling(
                per_lvl_feature, 
                [gt_box.bbox], 
                [q_feat],
                [center_per_img], 
                feat.unsqueeze(0)
            )

            # (n_gt, feat_dim)

            gt_box.add_field("feats", l2norm(reid_bn(box_feat)))
        
        return gt_boxes
                
    def forward(self, anchors, centers, objectness, box_regression, 
        box_offset, features, 
        reid_pool, targets, query=False, reid_feat=None, reid_bn=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
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

        for a, o, b, f in zip(anchors, objectness, box_regression, box_dconv_feat):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b, f, targets, query))

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
            
            # box_dconv_feat_flat_N: (N, H1W1+H2W2+..., dim)

            box_feat = self.reid_feat_func_pooling(features, 
                [boxlist.bbox for boxlist in boxlists], 
                points_feat, 
                centers,
                box_dconv_feat_flat_N)

            box_feat = box_feat.split(len_boxes)
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
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                feats_j = feats[inds]
                boxes_j = boxes[inds, :].view(-1, 4) 

                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class.add_field("points_feats", feats_j)
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


class RetinaNetPostProcessorFeatureAggregation(RetinaNetPostProcessor):
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
        super(RetinaNetPostProcessorFeatureAggregation, self).__init__(
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

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            feats = boxlists[i].get_field("points_feats")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                feats_j = feats[inds]
                boxes_j = boxes[inds, :].view(-1, 4)

                boxlist_for_class0 = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class0.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class0, self.nms_thresh,
                    score_field="scores"
                )

                iou_mat = boxlist_iou(boxlist_for_class0, boxlist_for_class)
                # (n_all, n_select)
                ovlp_topk = min(self.q_ovlp_topk, boxlist_for_class0.bbox.size(0))
                kthval, indices = torch.kthvalue(iou_mat, ovlp_topk, dim=0)
                max_val, max_indices = iou_mat.max(dim=0)
                mat1 = (iou_mat >= kthval)
                # mat2 = (iou_mat >= self.q_ovlp_thsh)
                mat2 = (iou_mat >= 0.6)
                thsh = (mat1*mat2).float()
                # (n_all, n_select)
                thsh[max_indices, torch.arange(boxlist_for_class.bbox.size(0))] = 1.0

                feat = feats_j

                norm = thsh.sum(dim=0)
                # (n_all, feat_dim)
                # feat = l2norm(feat)
                feat = torch.matmul(thsh.t(), feat)/norm.unsqueeze(-1)
                # feat = torch.matmul(thsh.t(), feat)# /norm.unsqueeze(-1)
                
                boxlist_for_class.add_field("points_feats", feat)          
                
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


def make_retinanet_postprocessor(config, rpn_box_coder, is_train):
    pre_nms_thresh = config.MODEL.RETINANET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.RETINANET.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.RETINANET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    min_size = 0

    query_ovlp_thsh = config.MODEL.PCNET.QUERY_OVERLAP_THSH
    query_ovlp_anc = config.MODEL.PCNET.QUERY_OVERLAP_ANCHOR
    query_ovlp_topk = config.MODEL.PCNET.QUERY_OVERLAP_TOPK

    if config.MODEL.REID_EVAL.DRAW_PCNET_POINTS:
        from .inference_drawpoints import RetinaNetPostProcessorDrawPoints
        postprocessor = RetinaNetPostProcessorDrawPoints
    elif config.MODEL.PCNET.GALLERY_AGGREGATION:
        postprocessor = RetinaNetPostProcessorFeatureAggregation
    else:
        postprocessor = RetinaNetPostProcessor

    box_selector = postprocessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=min_size,
        num_classes=config.MODEL.RETINANET.NUM_CLASSES,
        box_coder=rpn_box_coder,
        query_ovlp_thsh=query_ovlp_thsh,
        query_ovlp_anc=query_ovlp_anc,
        query_ovlp_topk=query_ovlp_topk,
    )

    return box_selector
