from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict
import os, cv2
from tqdm import tqdm

registry = {
    "PRW": ["maskrcnn_benchmark/datasets/PRW-v16.04.20/frames", "frame"],
    "CUHK-SYSU": ["maskrcnn_benchmark/datasets/CUHK-SYSU/dataset/Image/SSM", "t_list"], 
    "LSPS": None
}

def draw_bbox(dataset, predictions, output_folder, dataset_name='PRW'):

    frame_dir_path, frame_name_attr = registry[dataset_name]
    frames = getattr(dataset, frame_name_attr)

    draw_bbox_output_folder = os.path.join(output_folder, 'draw_bbox')
    if not os.path.exists(draw_bbox_output_folder):
        os.makedirs(draw_bbox_output_folder)

    n_draw_img = 1000

    print('Plotting the boxes under '+output_folder+'/draw_bbox...')
    for image_id, prediction in tqdm(enumerate(predictions)):

        # To process the whole dataset, just comment the 'if' statement below 
        if image_id == n_draw_img:
            break

        gt_bboxlist = dataset.get_groundtruth(image_id)
        img_info = dataset.get_img_info(image_id)

        gt_bbox = gt_bboxlist.bbox
        width = img_info["width"]
        height = img_info["height"]

        prediction = prediction.resize((width, height))
        pred_bbox = prediction.bbox

        frame_name = frames[image_id]

        img_name = os.path.join(frame_dir_path, frame_name)

        img = cv2.imread(img_name)

        for n in range(len(pred_bbox)):
            bbox = pred_bbox[n]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)

        cv2.imwrite(os.path.join(draw_bbox_output_folder, frame_name), img)




    return None

