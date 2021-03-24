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

def CUHK_train_test_split(datadir, istrain=True, gallery_size=100):
    anot_dir = os.path.join(datadir, 'annotation')
    frame_dir = os.path.join(datadir, 'Image/SSM')
    img_list = loadmat(os.path.join(anot_dir, 'Images.mat'))

    img_list = img_list[list(img_list.keys())[-1]]
    n_img = img_list.shape[1]
    mapp = defaultdict(np.array)
    for i in range(n_img):
        imgname, n_per = img_list[0, i][0][0], img_list[0, i][1][0][0]
        store_list = []
        for j in range(n_per):
            store_list.append(list(img_list[0, i][2][0][j][0][0]))
        nd_storelist = np.array(store_list)
        nd_storelist = np.column_stack((nd_storelist, -np.ones(nd_storelist.shape[0])))
        mapp[imgname] = np.array(nd_storelist)

    # separating the train and test frames
    print('Separating training and test set...')
    testlist = loadmat(os.path.join(anot_dir, 'pool.mat'))
    testlist = testlist[list(testlist.keys())[-1]]
    test_list = []
    for i in range(testlist.shape[0]):
        test_list.append(testlist[i][0][0])
    train_list = list(set(list(mapp.keys())) - set(test_list))

    print('Assigning the training id...')
    trainidmat = loadmat(os.path.join(anot_dir, 'test/train_test/Train.mat'))
    trainidmat = trainidmat[list(trainidmat.keys())[-1]]
    n_trainid = trainidmat.shape[0]

    idmapping = defaultdict(int)
    idmapp = []
    for i in range(n_trainid):
        idd = int(trainidmat[i][0][0][0][0][0][1:])
        idmapp.append(idd)
        # no need for testing phase
        n_appear = trainidmat[i][0][0][0][1][0][0]
        for k in range(n_appear):
            tmp = trainidmat[i][0][0][0][2][0][k]
            idimgname, loc = tmp[0][0], tmp[1]

            arr = mapp[idimgname]
            arr[np.argmin(np.linalg.norm(arr[:, :4] - loc, axis=1)), 4] = idd
    print('Done.')

    print('Remapping the original id...')
    idmapp = np.array(idmapp)
    idmapp_uniq = np.sort(np.unique(idmapp))
    idmapp_uniq = idmapp_uniq[idmapp_uniq>=0]
    for k in range(idmapp_uniq.shape[0]):
        idmapping[idmapp_uniq[k]] = k

    print('Done')

    print('Assigning the test id...')

    testidmat = loadmat(os.path.join(anot_dir, 'test/train_test/TestG100.mat'))
    testidmat = testidmat[list(testidmat.keys())[-1]]

    query_gallery_map = defaultdict(list)
    for j in range(testidmat[0].shape[0]):
        qgpair = testidmat[0][j]
        query_name = testidmat[0][j][0][0][0][0][0]
        query_coord = testidmat[0][j][0][0][0][1]
        tmp = []
        tmp.append(query_coord)

        query_id = int(testidmat[0][j][0][0][0][3][0][1:])
        tmp.append(query_id)
        gallery = testidmat[0][j][1][0]
        info = testidmat[0][j][1][0]
        arr = mapp[query_name]
        arr[np.argmin(np.linalg.norm(arr[:, :4] - query_coord, axis=1)), 4] = query_id
        for k in range(info.shape[0]):
            curr_info = info[k]
            coord = curr_info[1]
            gname = curr_info[0][0]
            tmp.append(gname)
            if coord.shape[1] == 0:
                continue
            arr = mapp[gname]
            arr[np.argmin(np.linalg.norm(arr[:, :4] - coord, axis=1)), 4] = query_id
        query_gallery_map[query_name].append(tmp)

    print('Done')

    print('xywh to xyxy coordinate...')

    for k, arr in mapp.items():
        arr[:, 2] = arr[:, 2] + arr[:, 0]
        arr[:, 3] = arr[:, 3] + arr[:, 1]

    for k, lst in query_gallery_map.items():
        for j in range(len(lst)):
            box_ = lst[j][0]
            box_ = box_.astype(np.int16)
            lst[j][0] = box_
            box_[:, 2] = box_[:, 0] + box_[:, 2]
            box_[:, 3] = box_[:, 1] + box_[:, 3]

    print('Done')

    return train_list, test_list, mapp, idmapping, query_gallery_map


class CUHKDataset_train():
    def __init__(self, ds_dir, transforms=None):
        global gtest_list, gmapp, glob_query_gallery_map

        gtrain_list, gtest_list, gmapp, gidmapping, glob_query_gallery_map = CUHK_train_test_split(ds_dir)

        self.transforms = transforms
        self.frame_dir = os.path.join(ds_dir, 'Image/SSM')
        self.mapp = gmapp
        
        self.t_list = gtrain_list
        
        self.idmapping = gidmapping

        self.name_shape_map = defaultdict(list)
        shapefile = os.path.join(ds_dir, 'train_shape.txt')
        f = open(shapefile)
        for line in f:
            sp = line.split('\t')
            sp[-1] = sp[-1][:-1]
            self.name_shape_map[sp[0]].extend([int(sp[1]), int(sp[2])])

        self.N_KP = 16

    def __len__(self):
        return len(self.t_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.frame_dir, self.t_list[idx])
        img = Image.open(img_name)
        
        boxlist = self.get_groundtruth(idx)
        boxlist = boxlist.clip_to_image(remove_empty=True)

        if self.transforms:
            image, boxlist = self.transforms(img, boxlist)

        return image, boxlist, idx

    def get_groundtruth(self, idx):
        bbox_raw = self.mapp[self.t_list[idx]]

        idd = bbox_raw[:, 4].astype(np.int32)
 
        for k in range(idd.shape[0]):
            if idd[k] < 0: continue
            idd[k] = self.idmapping[idd[k]]

        bbox = []
        for j in range(bbox_raw.shape[0]):
            bbox.append(bbox_raw[j, :4].tolist())

        label = torch.tensor([1]*idd.shape[0])
        idd = torch.tensor(idd)
        difficult = torch.tensor([False]*idd.shape[0])

        info = self.get_img_info(idx)

        keypoint_bbox = np.zeros([idd.shape[0], self.N_KP, 3]).tolist()
        keypoint_bbox = PersonKeypoints(keypoint_bbox, (info["width"], info["height"]))

        boxlist = BoxList(bbox, (info["width"], info["height"]), mode="xyxy")
        boxlist.add_field("labels", label)
        boxlist.add_field("ids", idd)
        boxlist.add_field("difficult", difficult)
        boxlist.add_field("keypoints", keypoint_bbox)

        return boxlist

    def get_img_info(self, idx):
        shape = self.name_shape_map[self.t_list[idx]]
        return {"height": shape[0], "width": shape[1]}


class CUHKDataset_test():
    def __init__(self, ds_dir, transforms=None):

        gtrain_list, gtest_list, gmapp, gidmapping, glob_query_gallery_map = CUHK_train_test_split(ds_dir)
        self.mapp = gmapp

        self.transforms = transforms
        self.frame_dir = os.path.join(ds_dir, 'Image/SSM')
 
        self.t_list = gtest_list
      
        self.name_shape_map = defaultdict(list)
        shapefile = os.path.join(ds_dir, 'test_shape.txt')
        f = open(shapefile)
        for line in f:
            sp = line.split('\t')
            sp[-1] = sp[-1][:-1]
            self.name_shape_map[sp[0]].extend([int(sp[1]), int(sp[2])])

        self.reverse_frame2idx_map = {}
        for j in range(len(self.t_list)):
            self.reverse_frame2idx_map[self.t_list[j]] = j

    def __len__(self):
        return len(self.t_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.frame_dir, self.t_list[idx])
        img = Image.open(img_name)
        
        boxlist = self.get_groundtruth(idx)
        boxlist = boxlist.clip_to_image(remove_empty=True)

        if self.transforms:
            image, boxlist = self.transforms(img, boxlist)

        return image, boxlist, idx

    def get_groundtruth(self, idx):
        bbox_raw = self.mapp[self.t_list[idx]]

        idd = bbox_raw[:, 4].astype(np.int32)

        bbox = []
        for j in range(bbox_raw.shape[0]):
            bbox.append(bbox_raw[j, :4].tolist())

        label = torch.tensor([1]*idd.shape[0])
        idd = torch.tensor(idd)
        difficult = torch.tensor([False]*idd.shape[0])

        info = self.get_img_info(idx)
        
        boxlist = BoxList(bbox, (info["width"], info["height"]), mode="xyxy")
        boxlist.add_field("labels", label)
        boxlist.add_field("ids", idd)
        boxlist.add_field("difficult", difficult)

        return boxlist

    def get_img_info(self, idx):
        shape = self.name_shape_map[self.t_list[idx]]
        return {"height": shape[0], "width": shape[1]}

    def eval_search_performance(self, dataset, predictions, qdataset, query_predictions, 
        eval_with_gt_gallery, plotting_options, 
        output_folder, logger):

        if plotting_options['draw_points']:
            from .visualization.draw_point import draw_point
            draw_point(dataset, predictions, output_folder, 'CUHK-SYSU')
            return
        elif plotting_options['draw_boxes']:
            from .visualization.draw_bbox import draw_bbox
            draw_bbox(dataset, predictions, output_folder, 'CUHK-SYSU')
            return

        det_thresh = 0.5
        name_to_det_feat = {}

        print('Processing name_to_det_feat...')
        for image_id, prediction in enumerate(predictions):
            name = dataset.t_list[image_id]
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

        q_feat, q_id, q_imgname = [], [], []
        print('FOWARD QUERY...')
        for image_id, qpred in enumerate(query_predictions):
            gt_bboxlist = qdataset.get_groundtruth(image_id)
            qids = qpred.get_field("ids")
            qfeat = qpred.get_field("feats")
            qimgname = qpred.get_field("imgname")
            
            q_feat.append(qfeat)
            q_id.extend(list(qids))
            q_imgname.extend(list(qimgname))

        q_feat = np.concatenate(q_feat, axis=0)
        q_id = np.array(q_id)
        q_imgname = np.array(q_imgname)

        aps, accs = [], []
        topk = [1, 5, 10]
        for i in tqdm(range(q_feat.shape[0])):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0

            feat_p = q_feat[i, :]
            probe_imgname = qdataset.frame[q_imgname[i]]
            
            probe_pid = q_id[i]

            probe_gts = {}

            q_counter_dataset = qdataset.qgmap_byidx[i]
        
            for gallery_name in q_counter_dataset:
                gallery_image_id = dataset.reverse_frame2idx_map[gallery_name]
                gt_bboxlist = dataset.get_groundtruth(gallery_image_id)

                gt_ids = gt_bboxlist.get_field("ids")
                if probe_pid in gt_ids and gallery_name != probe_imgname:
                    loc = np.where(gt_ids==probe_pid)[0]
                    probe_gts[gallery_name] = np.array(gt_bboxlist.bbox)[loc]
        
            for gallery_name in q_counter_dataset:
                if gallery_name == probe_imgname:
                    continue
                count_gt += (gallery_name in probe_gts)
                if gallery_name not in name_to_det_feat:
                    continue

                det, feat_g, pids_g = name_to_det_feat[gallery_name]
                sim = np.dot(feat_g, feat_p).ravel()
                label = np.zeros(len(sim), dtype=np.int32)

                if gallery_name in probe_gts:
                    gt = probe_gts[gallery_name].ravel()
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
        
        
class CUHKDataset_query():
    def __init__(self, ds_dir, transforms=None):
        gtrain_list, gtest_list, gmapp, gidmapping, glob_query_gallery_map = CUHK_train_test_split(ds_dir)
        query_gallery_map = glob_query_gallery_map

        self.mapp = defaultdict(list)
        self.frame = []

        self.qgmap_byidx = []

        for k, v in query_gallery_map.items():
            self.frame.append(k)
            for j in range(len(v)):
                tmp = []
                qg = v[j]
                coord = qg[0].tolist()[0]
                tmp.append(qg[1])
                tmp.extend(coord)
                self.mapp[k].append(tmp)

                self.qgmap_byidx.append(qg[2:])
                    
        self.transforms = transforms
        self.frame_dir = os.path.join(ds_dir, 'Image/SSM')
 
        self.name_shape_map = defaultdict(list)
        shapefile = os.path.join(ds_dir, 'test_shape.txt')
        f = open(shapefile)
        for line in f:
            sp = line.split('\t')
            sp[-1] = sp[-1][:-1]
            self.name_shape_map[sp[0]].extend([int(sp[1]), int(sp[2])])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.frame_dir, self.frame[idx])
        img = Image.open(img_name)
        
        boxlist = self.get_groundtruth(idx)
        boxlist = boxlist.clip_to_image(remove_empty=True)

        if self.transforms:
            image, boxlist = self.transforms(img, boxlist)

        return image, boxlist, idx

    def get_groundtruth(self, idx):
        bbox_arr = self.mapp[self.frame[idx]]
        bbox_arr = np.array(bbox_arr)

        bbox_raw = bbox_arr[:, 1:].astype(np.float32)
        bbox = []
        for j in range(bbox_raw.shape[0]):
            bbox.append(bbox_raw[j, :].tolist())

        idd = bbox_arr[:, 0].astype(np.int32)

        label = torch.tensor([1]*idd.shape[0])
        imgname = torch.tensor([idx]*idd.shape[0])

        difficult = torch.tensor([False]*idd.shape[0])

        idd = torch.tensor(idd)

        info = self.get_img_info(idx)
        
        boxlist = BoxList(bbox, (info["width"], info["height"]), mode="xyxy")
        boxlist.add_field("labels", label)
        boxlist.add_field("ids", idd)
        boxlist.add_field("difficult", difficult)
        boxlist.add_field("imgname", imgname)

        return boxlist

    def get_img_info(self, idx):
        shape = self.name_shape_map[self.frame[idx]]
        return {"height": shape[0], "width": shape[1]}


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




