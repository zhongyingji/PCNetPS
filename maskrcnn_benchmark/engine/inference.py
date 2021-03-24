# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str



# TODO
# merge compute_on_dataset and compute_on_dataset_query
def compute_on_dataset(model, data_loader, eval_with_gallery_gt, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    for _, batch in enumerate(tqdm(data_loader)):
        z += 1
        images, targets, image_ids = batch
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            if timer:
                timer.tic()

            # testing by groundtruth gallery boxes
            output = model(images, targets=targets if eval_with_gallery_gt else None, query=eval_with_gallery_gt)
            
            if timer:
                torch.cuda.synchronize()
                timer.toc()
       
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}

        )

    return results_dict


def compute_on_dataset_query(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    for _, batch in enumerate(tqdm(data_loader)):
        z += 1
        images, targets, image_ids = batch
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images, targets=targets, query=True)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
       
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )


    return results_dict



def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))


    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def inference_reid(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        q_dloader=None, 
        eval_with_gt_gallery=False, 
        plotting_options=None
):
    
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    qdataset = q_dloader.dataset

    query_forward = False
    if not (plotting_options["draw_points"] or plotting_options["draw_boxes"]):
        query_forward = True

    
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, eval_with_gt_gallery, device, inference_timer)
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if query_forward:
        total_timer.tic()
        query_predictions = compute_on_dataset_query(model, q_dloader, device, inference_timer)
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "[QUERY] Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(qdataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "[QUERY] Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(qdataset),
                num_devices,
            )
        )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    query_predictions = None if not query_forward else _accumulate_predictions_from_multiple_gpus(query_predictions)

    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        torch.save(query_predictions, os.path.join(output_folder, "query_predictions.pth"))

    dataset.eval_search_performance(dataset, predictions, qdataset, query_predictions, 
        eval_with_gt_gallery, plotting_options, 
        output_folder, logger=None)




    
