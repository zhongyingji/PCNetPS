import torch
import os
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

class LSPSDataset(torch.utils.data.Dataset):
    def __init__(self, ds_dir, mode='train', transforms=None):

        coord_info = os.path.join(ds_dir, mode+'_frame_coord.txt')
        id_info = os.path.join(ds_dir, mode+'_frame_id.txt')
        shape_info = os.path.join(ds_dir, mode+'_shape.txt')

        self.frame2coord = defaultdict(list)
        self.frame2id = defaultdict(list)

        print('Processing LSPS '+mode+' coordinate...')
        f = open(coord_info)
        for line in f:
            l = line.split('\t')
            l = l[:-1]
            for j in range(1, len(l), 4):
                self.frame2coord[l[0]].append([int(l[j]), int(l[j+1]), int(l[j+2]), int(l[j+3])])
        f.close()
        print('Processing LSPS '+mode+' id...')
        f = open(id_info)
        for line in f:
            l = line.split('\t')
            l[-1] = l[-1][:-1]
            for j in l[1:]:
                self.frame2id[l[0]].append(int(j))
        f.close()


        self.transforms = transforms
        self.frame_dir = os.path.join(ds_dir, mode)
        self.frame = os.listdir(self.frame_dir)
        self.n_frame = len(self.frame)
        self.mode = mode

        self.name_shape_map = defaultdict(list)

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

        print('Processing keypoint mapping...')
        self.kp_score_map, self.kp_pos_map = self._get_kp_map(ds_dir)
        print('Keypoing mapping done.')

        self.INVIS_THRESH = 0.1 # if the score lower than the threshold, it is invisible
        self.N_KP = 16


    def _get_kp_map(self, ds_dir):
        if self.mode == 'train':
            scores_path = 'scores.txt'
            pred_path = 'pred.txt'
        elif self.mode == "test":
            scores_path = 'scores_test.txt'
            pred_path = 'pred_test.txt'

        kp_score_path = os.path.join(ds_dir, scores_path)
        kp_pred_path = os.path.join(ds_dir, pred_path)
        f_score = open(kp_score_path)
        f_pred = open(kp_pred_path)

        score_map = defaultdict(list)
        pred_map = defaultdict(list)

        for line in f_score:
            a = line.split('\t')
            a[-1] = a[-1][:-1]
            tmp = []
            for k in a[1:]:
                tmp.append(float(k))
            score_map[a[0]].append(tmp)

        for line in f_pred:
            a = line.split('\t')
            a[-1] = a[-1][:-1]
            tmp = []
            for k in a[1:]:
                tmp.append(int(k))
            pred_map[a[0]].append(tmp)    
    
        f_score.close()
        f_pred.close()

        for k, v in score_map.items():
            l = len(v)
            if l > 1:
                #print(k)
                
                idx = 0
                conf = 0
                for j in range(l):
                    curr_conf = np.sum(v[j])
                    if curr_conf > conf:
                        conf = curr_conf
                        idx = j
                score_map[k] = [v[idx]]
                pred_map[k] = [pred_map[k][idx]]
        return score_map, pred_map


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

        bbox_coord = self.frame2coord[self.frame[idx]]
        bbox_id = self.frame2id[self.frame[idx]]
        bbox_raw = np.array(bbox_coord)
        bbox = []




        
        kp_loc = []
        kp_score = []

        pre = self.frame[idx].split('.')[0]

        

        for j in range(bbox_raw.shape[0]):

            #bbox_raw[j, 2] = bbox_raw[j, 0]+bbox_raw[j, 2]
            #bbox_raw[j, 3] = bbox_raw[j, 1]+bbox_raw[j, 3]
            bbox.append(bbox_raw[j, :].tolist())
                

        
            bbox_name = pre+'p_'+str(j+1)+'.jpg'

            xmin = max(bbox_raw[j, 0], 0)
            ymin = max(bbox_raw[j, 1], 0)
            if bbox_name not in self.kp_pos_map:
                kp_score.append([-1]*self.N_KP)
                kp_loc.append([-1]*self.N_KP*2)
                # sum score < 0
                # do not regress

            else:
                kp_score.append(self.kp_score_map[bbox_name][0])
                tmp_loc = self.kp_pos_map[bbox_name][0]
                tmp_loc = np.array(tmp_loc)
                tmp_loc[::2] += int(xmin)
                tmp_loc[1::2] += int(ymin)
                kp_loc.append(tmp_loc.tolist())




        kp_loc_ = np.array(kp_loc)
        kp_loc_ = np.reshape(kp_loc_, (bbox_raw.shape[0], self.N_KP, 2))
        kp_score_ = np.array(kp_score)
        nks, kks = np.where(kp_score_<self.INVIS_THRESH)
        kp_score_[kp_score_<self.INVIS_THRESH] = 0
        for nk, kk in zip(nks, kks):
            kp_loc_[nk, kk, :] = 0
        keypoint_bbox = np.concatenate((kp_loc_, kp_score_[:, :, None]), axis=2).tolist()

        
        
        idd = np.array(bbox_id, dtype=np.int32)

        label = torch.tensor([1]*idd.shape[0])
        # check if it is 0 or 1

        difficult = torch.tensor([False]*idd.shape[0])
        
        cam = int(self.frame[idx].split('_')[2])
        cam = torch.tensor([cam]*idd.shape[0])

        img_name = [self.frame[idx]]*idd.shape[0]


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

        for k, v in self.frame2id.items():
            self.idlist.extend(v)

        uniq = np.unique(np.array(self.idlist))
        uniq = uniq[uniq >= 0]
        sort_uniq = np.sort(uniq)
        for k in range(sort_uniq.shape[0]):
            self.idmap[sort_uniq[k]] = k


        return


class LSPSQuery(torch.utils.data.Dataset):
    def __init__(self, ds_dir, transforms=None):
        path = os.path.join(ds_dir, "query_info.txt")
        self.frame_dir = os.path.join(ds_dir, "test/")
      
        self.mapping = defaultdict(list)
        self.frame = []


        f = open(path)
        for line in f:
            z = line.split('\t')
            z[-1] = z[-1][:-1]
            tmp = []
            tmp.append(int(z[0]))
            for k in z[2:]:
                tmp.append(float(k))
            self.mapping[z[1]].append(tmp)

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
        f.close()

        self.frame2id = defaultdict(list)
        id_info = os.path.join(ds_dir, 'test_frame_id.txt')
        print('Processing LSPS test id...')
        f = open(id_info)
        for line in f:
            l = line.split('\t')
            l[-1] = l[-1][:-1]
            for j in l[1:]:
                self.frame2id[l[0]].append(int(j))
        f.close()



        print('Processing keypoint mapping...')
        self.kp_score_map, self.kp_pos_map = self._get_kp_map(ds_dir)
        print('Keypoint mapping done.')

        self.N_KP = 16
        self.INVIS_THRESH = 0.1


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
        idlist_inframe = np.array(self.frame2id[self.frame[idx]])
        
        bbox_raw = bbox_arr[:, 1:].astype(np.float32)
        #bbox_raw = bbox_raw.tolist()
        bbox = []
        kp_score = []
        kp_loc = []

        

        pre = self.frame[idx].split('.')[0]

        

        for j in range(bbox_raw.shape[0]):
            #bbox_raw[j, 2] = bbox_raw[j, 0]+bbox_raw[j, 2]
            #bbox_raw[j, 3] = bbox_raw[j, 1]+bbox_raw[j, 3]
            bbox.append(bbox_raw[j, :].tolist())

        

            _idx = np.where(idlist_inframe==int(bbox_arr[j, 0]))[0][0]
            kp_box_name = pre+'p_'+str(_idx+1)+'.jpg'
            xmin = max(bbox_raw[j, 0], 0)
            ymin = max(bbox_raw[j, 1], 0)
            
            if kp_box_name not in self.kp_pos_map:
                kp_score.append([-1]*self.N_KP)
                kp_loc.append([-1]*self.N_KP*2)
                # sum score < 0
                # do not regress

            else:
                kp_score.append(self.kp_score_map[kp_box_name][0])
                tmp_loc = self.kp_pos_map[kp_box_name][0]
                tmp_loc = np.array(tmp_loc)
                tmp_loc[::2] += int(xmin)
                tmp_loc[1::2] += int(ymin)
                kp_loc.append(tmp_loc.tolist())


        kp_loc_ = np.array(kp_loc)
        kp_loc_ = np.reshape(kp_loc_, (bbox_raw.shape[0], self.N_KP, 2))
        kp_score_ = np.array(kp_score)
        nks, kks = np.where(kp_score_<self.INVIS_THRESH)
        kp_score_[kp_score_<self.INVIS_THRESH] = 0
        for nk, kk in zip(nks, kks):
            kp_loc_[nk, kk, :] = 0
        keypoint_bbox = np.concatenate((kp_loc_, kp_score_[:, :, None]), axis=2).tolist()



        


        idd = bbox_arr[:, 0].astype(np.int32)

        label = torch.tensor([1]*idd.shape[0])

        difficult = torch.tensor([False]*idd.shape[0])
        # check if it is 0 or 1

        idd = torch.tensor(idd)

        info = self.get_img_info(idx)
        
        cam = int(self.frame[idx].split('_')[2])
        cam = torch.tensor([cam]*idd.shape[0])

        imgname = torch.tensor([idx]*idd.shape[0])


        
        keypoint_bbox = PersonKeypoints(keypoint_bbox, (info["width"], info["height"]))

        
        boxlist = BoxList(bbox, (info["width"], info["height"]), mode="xyxy")
        boxlist.add_field("labels", label)
        boxlist.add_field("ids", idd)
        boxlist.add_field("cams", cam)
        boxlist.add_field("difficult", difficult)
        boxlist.add_field("keypoints", keypoint_bbox)
        boxlist.add_field("imgname", imgname)

        return boxlist

    def get_img_info(self, idx):
        shape = self.name_shape_map[self.frame[idx]]
        return {"height": shape[0], "width": shape[1]}


    def _get_kp_map(self, ds_dir):
        kp_score_path = os.path.join(ds_dir, 'scores_test.txt')
        kp_pred_path = os.path.join(ds_dir, 'pred_test.txt')
        f_score = open(kp_score_path)
        f_pred = open(kp_pred_path)

        score_map = defaultdict(list)
        pred_map = defaultdict(list)

        for line in f_score:
            a = line.split('\t')
            a[-1] = a[-1][:-1]
            tmp = []
            for k in a[1:]:
                tmp.append(float(k))
            score_map[a[0]].append(tmp)

        for line in f_pred:
            a = line.split('\t')
            a[-1] = a[-1][:-1]
            tmp = []
            for k in a[1:]:
                tmp.append(int(k))
            pred_map[a[0]].append(tmp)    
    
        f_score.close()
        f_pred.close()

        for k, v in score_map.items():
            l = len(v)
            if l > 1:
                #print(k)
                
                idx = 0
                conf = 0
                for j in range(l):
                    curr_conf = np.sum(v[j])
                    if curr_conf > conf:
                        conf = curr_conf
                        idx = j
                score_map[k] = [v[idx]]
                pred_map[k] = [pred_map[k][idx]]
        return score_map, pred_map

