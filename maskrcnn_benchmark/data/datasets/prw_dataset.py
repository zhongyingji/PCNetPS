import torch
import os
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

from sklearn.metrics import average_precision_score
from tqdm import tqdm

class PRWDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "person", 
    )

    def __init__(self, ds_dir, mode='train', transforms=None):
        path = os.path.join(ds_dir, 'frame_' + mode + '.mat')
        frame = loadmat(path)
        frame = frame['img_index_' + mode]

        self.transforms = transforms
        self.frame_dir = os.path.join(ds_dir, "frames")
        self.ann_dir = os.path.join(ds_dir, "annotations")

        self.name_shape_map = defaultdict(list)

        self.mode = mode
        self.n_frame = frame.shape[0]
        self.frame = []

        for i in range(self.n_frame):
            self.frame.append(frame[i][0][0] + '.jpg')

        shapefile = os.path.join(ds_dir, mode+'_shape.txt')
        f = open(shapefile)
        for line in f:
            sp = line.split('\t')
            sp[-1] = sp[-1][:-1]
            self.name_shape_map[sp[0]].extend([int(sp[1]), int(sp[2])])

        if self.mode=='train':
            print('Processing id mapping...')
            self._process_idmap()
            print('Processing done.')

        kp_score_path = os.path.join(ds_dir, 'scores.txt')
        kp_pos_path = os.path.join(ds_dir, 'pred.txt')
        self.kp_score_map = defaultdict(list)
        self.kp_pos_map = defaultdict(list)

        fs = open(kp_score_path)
        for line in fs:
            tmp = line.split('\t')
            tmp[-1] = tmp[-1][:-1]
            tmp_score = []
            for j in tmp[1:]:
                tmp_score.append(float(j))
            self.kp_score_map[tmp[0]].append(tmp_score)
        fs.close()

        fp = open(kp_pos_path)
        for line in fp:
            tmp = line.split('\t')
            tmp[-1] = tmp[-1][:-1]
            tmp_pos = []
            for j in tmp[1:]:
                tmp_pos.append(int(j))
            self.kp_pos_map[tmp[0]].append(tmp_pos)
        fp.close()

        self.INVIS_THRESH = 0.1 # if the score lower than the threshold, it is invisible
        self.N_KP = 16

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.frame_dir, self.frame[idx])

        image = Image.open(img_path)
        
        boxlist = self.get_groundtruth(idx)
        boxlist = boxlist.clip_to_image(remove_empty=True)
        
        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        #boxlist.add_field("ids", idd)
        return image, boxlist, idx

    def __len__(self):
        return self.n_frame

    def get_groundtruth(self, idx):
        bbox_path = os.path.join(self.ann_dir, self.frame[idx])
        bbox_arr = loadmat(bbox_path)
        bbox_arr = bbox_arr[list(bbox_arr.keys())[-1]]
        bbox_raw = bbox_arr[:, 1:].astype(np.float32)

        bbox = []

        for j in range(bbox_raw.shape[0]):

            bbox_raw[j, 2] = bbox_raw[j, 0]+bbox_raw[j, 2]
            bbox_raw[j, 3] = bbox_raw[j, 1]+bbox_raw[j, 3]
            bbox.append(bbox_raw[j, :].tolist())
            # list
        
        keypoint_bbox = []
        if self.kp_pos_map.__contains__(self.frame[idx]):

            kp_pos = np.array(self.kp_pos_map[self.frame[idx]])
            kp_posx = kp_pos[:, ::2]
            kp_posy = kp_pos[:, 1::2]
            kp_lx = np.min(kp_posx, axis=1, keepdims=True)
            kp_rx = np.max(kp_posx, axis=1, keepdims=True)
            kp_uy = np.min(kp_posy, axis=1, keepdims=True)
            kp_dy = np.max(kp_posy, axis=1, keepdims=True)
            kp_bbox = np.concatenate([kp_lx, kp_uy, kp_rx, kp_dy], axis=1)

            kp_score = np.array(self.kp_score_map[self.frame[idx]])
           
            # (x, y, x, y)
            np_bbox = torch.as_tensor(bbox, dtype=torch.float32)
            kp_bbox = torch.as_tensor(kp_bbox, dtype=torch.float32)

            area_bbox = (np_bbox[:, 2]-np_bbox[:, 0]+1) * (np_bbox[:, 3]-np_bbox[:, 1]+1)
            area_kpbbox = (kp_bbox[:, 2]-kp_bbox[:, 0]+1) * (kp_bbox[:, 3]-kp_bbox[:, 1]+1)

            lt = torch.max(np_bbox[:, None, :2], kp_bbox[:, :2])  # [N,M,2]
            rb = torch.min(np_bbox[:, None, 2:], kp_bbox[:, 2:])  # [N,M,2]
            TO_REMOVE = 1
            wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
            inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
            iou = inter / (area_bbox[:, None] + area_kpbbox - inter)

            bbox_kp_iou = np.array(iou)
           
            meanx = (bbox_raw[:, 0]+bbox_raw[:, 2])/2
            meany = (bbox_raw[:, 1]+bbox_raw[:, 3])/2
           
            selc_kp = np.argmax(bbox_kp_iou, axis=1)
            
            kbox_or_not = (kp_lx[selc_kp].squeeze() <= meanx) & (kp_rx[selc_kp].squeeze() >= meanx) & (
                kp_uy[selc_kp].squeeze() <= meany) & (kp_dy[selc_kp].squeeze() >= meany)
            selc_kp[~kbox_or_not] = -1
            
            for j in range(bbox_raw.shape[0]):
                if selc_kp[j] == -1:
                    tmpp = np.zeros((kp_pos.shape[1]//2 ,3))
                    keypoint_bbox.append(tmpp.tolist())
                    continue
                tmpx = kp_posx[selc_kp[j], :][:, None]
                tmpy = kp_posy[selc_kp[j], :][:, None]
                conf = kp_score[selc_kp[j]][:, None]
                invis = conf<self.INVIS_THRESH
                conf[invis] = 0
                tmpx[invis] = 0
                tmpy[invis] = 0
                keypoint_bbox.append(np.concatenate((tmpx, tmpy, conf), axis=1).tolist())
        else:
            keypoint_bbox = np.zeros((bbox_raw.shape[0], self.N_KP, 3), dtype=np.int32).tolist()

        
        idd = bbox_arr[:, 0].astype(np.int32)

        label = torch.tensor([1]*idd.shape[0])

        difficult = torch.tensor([False]*idd.shape[0])
        # check if it is 0 or 1
        cam = int(self.frame[idx][1])
        cam = torch.tensor([cam]*idd.shape[0])


        if self.mode == 'train':
            for k in range(idd.shape[0]):
                if idd[k] < 0: 
                    #id[k] = 5555
                    continue
                idd[k] = self.idmap[idd[k]]
        idd = torch.tensor(idd)

        info = self.get_img_info(idx)

        keypoint_bbox = PersonKeypoints(keypoint_bbox, (info["width"], info["height"]))


        boxlist = BoxList(bbox, (info["width"], info["height"]), mode="xyxy")
        boxlist.add_field("labels", label)
        boxlist.add_field("ids", idd)
        boxlist.add_field("cams", cam)
        boxlist.add_field("difficult", difficult)
        boxlist.add_field("keypoints", keypoint_bbox)

        return boxlist

    def get_img_info(self, idx):
        shape = self.name_shape_map[self.frame[idx]]
        return {"height": shape[0], "width": shape[1]}

    def _process_idmap(self):

        self.idmap = defaultdict(int)
        self.idlist = []
        for i in range(self.n_frame):
            bbox_path = os.path.join(self.ann_dir, self.frame[i])
            bbox_arr = loadmat(bbox_path)
            bbox_arr = bbox_arr[list(bbox_arr.keys())[-1]][:, 0].astype(np.int32)

            self.idlist.extend(list(bbox_arr))
        uniq = np.unique(np.array(self.idlist))
        uniq = uniq[uniq >= 0]
        sort_uniq = np.sort(uniq)
        for k in range(sort_uniq.shape[0]):
            self.idmap[sort_uniq[k]] = k

        return

    def map_class_id_to_class_name(self, class_id):
        return PRWDataset.CLASSES[class_id]

    def eval_search_performance(self, dataset, predictions, qdataset, query_predictions, 
        eval_with_gt_gallery, plotting_options, 
        output_folder, logger):

        assert not self.mode == 'train'

        if plotting_options['draw_points']:
            from .visualization.draw_point import draw_point
            draw_point(dataset, predictions, output_folder, 'PRW')
            return
        elif plotting_options['draw_boxes']:
            from .visualization.draw_bbox import draw_bbox
            draw_bbox(dataset, predictions, output_folder, 'PRW')
            return

        det_thresh = 0.5
        name_to_det_feat = {}

        print('Processing name_to_det_feat...')
        for image_id, prediction in enumerate(predictions):
            name = dataset.frame[image_id]
            gt_bboxlist = dataset.get_groundtruth(image_id)
            img_info = dataset.get_img_info(image_id)
            width = img_info['width']
            height = img_info['height']

            prediction = prediction.resize((width, height))
            det = np.array(prediction.bbox)
            det_feat = prediction.get_field("feats")
            det_feat = np.array(det_feat)
        
            pids = np.array(gt_bboxlist.get_field("ids"))

            if not eval_with_gt_gallery:
                scores = np.array(prediction.get_field("scores"))
            else:
                # for testing performance with groundtruth gallery boxes
                scores = np.ones(det.shape[0])

            inds = np.where(scores>=det_thresh)[0]

            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds, :], det_feat[inds, :], pids)

        q_feat, q_id, q_imgname, q_cam = [], [], [], []
        print('FOWARD QUERY...')
        for image_id, qpred in enumerate(query_predictions):
            gt_bboxlist = qdataset.get_groundtruth(image_id)

            qids = qpred.get_field("ids")
            qfeat = qpred.get_field("feats")
            qcam = qpred.get_field("cams")
            qimgname = qpred.get_field("imgname")
            
            q_feat.append(qfeat)
            q_id.extend(list(qids))
            q_imgname.extend(list(qimgname))
            q_cam.extend(list(qcam))

        q_feat = np.concatenate(q_feat, axis=0)
        q_id = np.array(q_id)
        q_imgname = np.array(q_imgname)
        q_cam = np.array(q_cam)

        aps, accs = [], []
        topk = [1, 5, 10]
        for i in tqdm(range(q_feat.shape[0])):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0

            feat_p = q_feat[i, :]
            probe_imgname = qdataset.frame[q_imgname[i]]

            probe_pid = q_id[i]
            probe_cam = q_cam[i]

            probe_gts = {}

            construct_gallery_imgname = []

            for image_id in range(len(dataset)):
                gt_bboxlist = dataset.get_groundtruth(image_id)
                name = dataset.frame[image_id]
                gt_ids = gt_bboxlist.get_field("ids")
                gt_cams = gt_bboxlist.get_field("cams")
                
                if probe_pid in gt_ids and name != probe_imgname:
                    loc = np.where(gt_ids==probe_pid)[0]
                    probe_gts[name] = np.array(gt_bboxlist.bbox)[loc]

            for image_id in range(len(dataset)):
                gallery_imgname = dataset.frame[image_id]
                if gallery_imgname == probe_imgname:
                    continue
                count_gt += (gallery_imgname in probe_gts)
                if gallery_imgname not in name_to_det_feat:
                    continue

                det, feat_g, pids_g = name_to_det_feat[gallery_imgname]
                sim = np.dot(feat_g, feat_p).ravel()
                label = np.zeros(len(sim), dtype=np.int32)

                if gallery_imgname in probe_gts:
                    gt = probe_gts[gallery_imgname].ravel()
                    w, h = gt[2] - gt[0], gt[3] - gt[1]

                    iou_thresh = min(0.5, (w*h*1.0)/((w+10)*(h+10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]

                    for j, roi in enumerate(det[:, :]):
                        if compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break

                y_true.extend(list(label))
                y_score.extend(list(sim))

            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            recall_rate = count_tp*1.0/count_gt
            ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score)*recall_rate
            aps.append(ap)

            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])

        mAP = np.mean(aps)
        accs_ = np.mean(accs, axis=0)
        print('mAP: {:.2%}'.format(mAP))
        for i, k in enumerate(topk):
            print('top-{:2d} = {:.2%}'.format(k, accs_[i]))

        
def compute_iou(box1, box2):
    # (4, )
    # (xmin, ymin, xmax, ymax)
    w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if w <= 0 or h <= 0:
        return 0
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    cross = w*h
    return cross/(area1+area2-cross)


class PRWQuery(torch.utils.data.Dataset):
    def __init__(self, ds_dir, transforms=None):
        path = os.path.join(ds_dir, "query_info.txt")
        self.frame_dir = os.path.join(ds_dir, "frames")
      
        self.mapping = defaultdict(list)
        self.frame = []

        f = open(path)
        for line in f:
            z = line.split(' ')
            z[-1] = z[-1][:-1]+'.jpg'
            tmp = []
            tmp.append(int(z[0]))
            for k in z[1:-1]:
                tmp.append(float(k))
            self.mapping[z[-1]].append(tmp)

        f.close()
        for k, v in self.mapping.items():
            self.frame.append(k)

        self.transforms = transforms

        self.name_shape_map = defaultdict(list)
        shapefile = os.path.join(ds_dir, 'test_shape.txt')
        f = open(shapefile)
        for line in f:
            sp = line.split('\t')
            sp[-1] = sp[-1][:-1]
            self.name_shape_map[sp[0]].extend([int(sp[1]), int(sp[2])])

    def __getitem__(self, idx):
        img_path = os.path.join(self.frame_dir, self.frame[idx])

        image = Image.open(img_path)
        
        boxlist = self.get_groundtruth(idx)
        boxlist = boxlist.clip_to_image(remove_empty=True)
        
        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        #boxlist.add_field("ids", idd)
        return image, boxlist, idx

    def __len__(self):
        return len(self.frame)

    def get_groundtruth(self, idx):

        bbox_arr = self.mapping[self.frame[idx]]
        bbox_arr = np.array(bbox_arr)
        bbox_raw = bbox_arr[:, 1:].astype(np.float32)
        #bbox_raw = bbox_raw.tolist()
        bbox = []
        for j in range(bbox_raw.shape[0]):
            bbox_raw[j, 2] = bbox_raw[j, 0]+bbox_raw[j, 2]
            bbox_raw[j, 3] = bbox_raw[j, 1]+bbox_raw[j, 3]
            bbox.append(bbox_raw[j, :].tolist())

        idd = bbox_arr[:, 0].astype(np.int32)

        label = torch.tensor([1]*idd.shape[0])
        imgname = torch.tensor([idx]*idd.shape[0])

        difficult = torch.tensor([False]*idd.shape[0])
        # check if it is 0 or 1

        idd = torch.tensor(idd)

        info = self.get_img_info(idx)
        
        cam = int(self.frame[idx][1])
        cam = torch.tensor([cam]*idd.shape[0])

        boxlist = BoxList(bbox, (info["width"], info["height"]), mode="xyxy")
        boxlist.add_field("labels", label)
        boxlist.add_field("ids", idd)
        boxlist.add_field("cams", cam)
        boxlist.add_field("difficult", difficult)
        boxlist.add_field("imgname", imgname)

        return boxlist

    def get_img_info(self, idx):
        shape = self.name_shape_map[self.frame[idx]]
        return {"height": shape[0], "width": shape[1]}

