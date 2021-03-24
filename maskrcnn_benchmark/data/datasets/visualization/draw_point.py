from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict
import os, cv2
from tqdm import tqdm


# Only supports PCNet

registry = {
    "PRW": ["maskrcnn_benchmark/datasets/PRW-v16.04.20/frames", "frame"],
    "CUHK-SYSU": ["maskrcnn_benchmark/datasets/CUHK-SYSU/dataset/Image/SSM", "t_list"], 
    "LSPS": None
}

def draw_point(dataset, predictions, output_folder, dataset_name='PRW'):

    frame_dir_path, frame_name_attr = registry[dataset_name]
    frames = getattr(dataset, frame_name_attr)

    draw_points_output_folder = os.path.join(output_folder, 'draw_points')
    if not os.path.exists(draw_points_output_folder):
        os.makedirs(draw_points_output_folder)

    color = (
        (0, 0, 255),
        (0, 255, 0), 
        (255, 0, 0),
        (0, 0, 0), 
        (255, 0, 255), 
        (255, 255, 0), 
        (18, 153, 255),
        (84, 46, 8), 
        (132, 227, 255), 
        (42, 42, 128)

    )

    n_draw_img = 10

    print('Plotting the points under '+output_folder+'/draw_points...')
    for image_id, prediction in tqdm(enumerate(predictions)):
        
        gt_bboxlist = dataset.get_groundtruth(image_id)
        img_info = dataset.get_img_info(image_id)

        gt_bbox = gt_bboxlist.bbox
        width = img_info["width"]
        height = img_info["height"]

        w, h = prediction.size

        pred_bbox = prediction.bbox

        ctrs = prediction.get_field("centers")
        offsets = prediction.get_field("offsets") # (n_pred, n_points*2)
        n_points = offsets.shape[1] // 2
        lvls = prediction.get_field("fpn_lvls")

        frame_name = frames[image_id]

        img_name = os.path.join(frame_dir_path, frame_name)
        img = cv2.imread(img_name)
        img = cv2.resize(img, (w, h))

        for n in range(len(pred_bbox)):
            bbox = pred_bbox[n]
            if n >= len(color):
                per_color = color[-1]
            else:
                per_color = color[n]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), per_color, 4)

            per_lvl = int(lvls[n])
            per_ctr_y, per_ctr_x = ctrs[n]
            per_ofs = offsets[n] 
            per_ofs *= (2**per_lvl)
            for nrp in range(n_points):
                x = per_ctr_x + per_ofs[2*nrp+1]
                y = per_ctr_y + per_ofs[2*nrp]
                cv2.circle(img, (x, y), 1, per_color, 4)
        
        cv2.imwrite(os.path.join(draw_points_output_folder, frame_name), img)

    return None

